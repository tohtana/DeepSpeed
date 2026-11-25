# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from datetime import timedelta
from hashlib import sha256
import multiprocessing as mp
import os
import signal
import socket

import pytest
import torch

import deepspeed
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.constants import FP32_FLAT_GROUPS, OPTIMIZER_STATE_DICT
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.op_builder import FusedAdamBuilder

FAILED_RANK = 1
SNAPSHOT_STEP = 2
TIMEOUT = 90


def _new_engine(rank, port, recovering=False):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    accelerator = get_accelerator()
    accelerator.set_device(rank)
    dist.init_distributed(timeout=timedelta(seconds=TIMEOUT))
    if recovering:
        deepspeed.zero.prepare_optimizer_state_recovery()
    torch.manual_seed(41)
    model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Tanh(), torch.nn.Linear(4, 4))
    optimizer_type = FusedAdam if accelerator.device_name() == "cuda" else torch.optim.AdamW
    optimizer = optimizer_type([{
        "params": model[0].parameters()
    }, {
        "params": model[2].parameters(),
        "lr": 0.03
    }],
                               lr=0.01)
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": False
        },
        "zero_allow_untested_optimizer": True,
    }
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
    return engine


def _step(engine, rank, step):
    for micro_batch in range(engine.gradient_accumulation_steps()):
        data = torch.arange(8, dtype=torch.float32) * (rank + 1) + micro_batch - step
        x = data.reshape(1, -1).to(device=engine.device, dtype=torch.bfloat16) / 16
        out = engine(x).float()
        engine.backward((out - (rank - step) / 8).square().mean())
        engine.step()


def _digest(tensors):
    digest = sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state(engine, step):
    shard = engine.optimizer.state_dict()
    masters = shard[FP32_FLAT_GROUPS]
    states = list(shard[OPTIMIZER_STATE_DICT]["state"].values())
    assert masters and all(t.dtype == torch.float32 for t in masters)
    assert states and all(int(s["step"]) == step for s in states)
    assert all(s[key].dtype == torch.float32 for s in states for key in ("exp_avg", "exp_avg_sq"))
    assert any(torch.any(t != t.bfloat16().float()) for t in masters)
    return {
        "masters": _digest(masters),
        "exp_avg": _digest(s["exp_avg"] for s in states),
        "exp_avg_sq": _digest(s["exp_avg_sq"] for s in states),
        "step": _digest(torch.as_tensor(s["step"]) for s in states),
        "model": _digest(p.ds_tensor for p in engine.module.parameters()),
    }


def _replica_group(rank):
    for first_rank in range(0, dist.get_world_size(), 2):
        group = dist.new_group([first_rank, first_rank + 1], backend="gloo")
        if first_rank <= rank < first_rank + 2:
            replica_group = group
    return replica_group


def _recover(rank, engine, local, peer):
    group = _replica_group(rank)
    if rank in (0, FAILED_RANK):
        payload = [peer if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0, group=group)
        if rank == FAILED_RANK:
            local = payload[0]

    with pytest.raises(ValueError, match="matching optimizer snapshot"):
        deepspeed.zero.restore_optimizer_state(engine, None if rank == FAILED_RANK else local)
    deepspeed.zero.restore_optimizer_state(engine, local)
    assert local.step == SNAPSHOT_STEP
    recovered = _state(engine, SNAPSHOT_STEP)
    _step(engine, rank, SNAPSHOT_STEP + 1)
    updated = _state(engine, SNAPSHOT_STEP + 1)
    engine.destroy()
    dist.destroy_process_group()
    return os.getpid(), recovered, updated


def _worker(rank, connection, port, replacement):
    engine = _new_engine(rank, port, recovering=replacement)
    if replacement:
        connection.send(_recover(rank, engine, None, None))
        return

    for step in range(1, SNAPSHOT_STEP + 1):
        _step(engine, rank, step)
    expected_before = _state(engine, SNAPSHOT_STEP)
    _step(engine, rank, SNAPSHOT_STEP + 1)
    expected_after = _state(engine, SNAPSHOT_STEP + 1)
    assert expected_before["masters"] != expected_after["masters"]
    engine.destroy()

    engine = _new_engine(rank, port)
    for step in range(1, SNAPSHOT_STEP + 1):
        _step(engine, rank, step)
    assert _state(engine, SNAPSHOT_STEP) == expected_before
    local, peer = deepspeed.zero.replicate_optimizer_state(engine, _replica_group(rank))
    assert (local.rank, peer.rank) == (rank, rank ^ 1)
    _step(engine, rank, SNAPSHOT_STEP + 1)
    assert _state(engine, SNAPSHOT_STEP + 1) == expected_after
    dist.barrier()
    connection.send((os.getpid(), expected_before, expected_after))

    port = connection.recv()
    engine.destroy()
    dist.destroy_process_group()
    engine = _new_engine(rank, port, recovering=True)
    connection.send(_recover(rank, engine, local, peer))


def _port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _receive(connection):
    assert connection.poll(TIMEOUT), "Timed out waiting for the training worker."
    return connection.recv()


@pytest.mark.skipif(os.name != "posix", reason="requires terminating a training worker with SIGKILL")
@pytest.mark.parametrize("accelerator", ["cpu", "cuda"])
def test_zero3_optimizer_replica_survives_worker_replacement(monkeypatch, accelerator):
    if accelerator == "cuda":
        actual = get_accelerator()
        if actual.device_name() != "cuda" or actual.device_count() < 2 or not actual.is_bf16_supported():
            pytest.skip("requires two CUDA devices with BF16 support")
        if not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME]:
            pytest.skip("requires FusedAdam")
        FusedAdamBuilder().load()
    world_size = 4 if accelerator == "cpu" else 2
    monkeypatch.setenv("DS_ACCELERATOR", accelerator)
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("LOCAL_SIZE", str(world_size))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    context = mp.get_context("spawn")
    workers = []

    def start(rank, port, replacement=False):
        parent, child = context.Pipe()
        process = context.Process(target=_worker, args=(rank, child, port, replacement))
        process.start()
        child.close()
        workers.append((process, parent))

    try:
        first_port = _port()
        for rank in range(world_size):
            start(rank, first_port)
        reference = [_receive(connection) for _, connection in workers]
        original, _ = workers[FAILED_RANK]
        original.kill()
        original.join(TIMEOUT)
        assert original.exitcode == -signal.SIGKILL

        next_port = _port()
        for rank, (_, connection) in enumerate(workers):
            if rank != FAILED_RANK:
                connection.send(next_port)
        start(FAILED_RANK, next_port, replacement=True)
        for rank in range(world_size):
            worker = workers[-1] if rank == FAILED_RANK else workers[rank]
            process, connection = worker
            pid, recovered, updated = _receive(connection)
            before_pid, before, after = reference[rank]
            assert pid == process.pid
            assert (before_pid != pid) == (rank == FAILED_RANK)
            assert recovered == before
            assert updated == after
            process.join(TIMEOUT)
            assert process.exitcode == 0
        print(f"SIGKILL rank {FAILED_RANK} PID {original.pid}. Replacement PID {workers[-1][0].pid}. "
              f"Survivors {[reference[rank][0] for rank in range(world_size) if rank != FAILED_RANK]}. "
              "FP32 masters, Adam moments and step, and BF16 partitions match before and after the next update.")
    finally:
        for process, connection in workers:
            connection.close()
            if process.is_alive():
                process.kill()
            process.join(TIMEOUT)
