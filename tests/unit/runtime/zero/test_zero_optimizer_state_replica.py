# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.utils import view_as_uint8
from unit.common import DistributedTest, reduce_boolean_flags


def _value_and_bits(value):
    if torch.is_tensor(value):
        bits = view_as_uint8(value.detach().cpu()).numpy().tobytes()
        return (value.dtype, value.shape, bits)
    return (type(value), value)


def _local_optimizer_state(engine, *, include_padding=True):
    zero = engine.optimizer
    masters = zero.fp32_partitioned_groups_flat
    states = [zero.optimizer.state[master] for master in masters]
    fields = (masters, [state["exp_avg"] for state in states], [state["exp_avg_sq"] for state in states])
    if not include_padding:
        fields = tuple([torch.cat(parts) for parts in zero.get_groups_without_padding(field)] for field in fields)
    steps = [state["step"] for state in states]
    return tuple(tuple(_value_and_bits(value) for value in subgroup) for subgroup in zip(*fields, steps))


def _replica_state(replica):
    for partition in replica.subgroups:
        for tensor in (partition.fp32_master, partition.exp_avg, partition.exp_avg_sq, partition.step):
            if torch.is_tensor(tensor):
                assert tensor.device.type == "cpu" and not tensor.requires_grad
    return tuple(
        tuple(
            _value_and_bits(value)
            for value in (partition.fp32_master, partition.exp_avg, partition.exp_avg_sq, partition.step))
        for partition in replica.subgroups)


def _train(engine, micro_step, overflow=False):
    dtype = torch.float16 if engine.fp16_enabled() else torch.float32
    inputs = torch.arange(15, device=engine.device, dtype=dtype).reshape(3, 5)
    inputs = inputs / 17 + dist.get_rank() / 7 + micro_step / 19
    loss = engine(inputs).float().square().mean()
    if overflow:
        loss = loss * float("inf")
    engine.backward(loss)
    engine.step()


def _make_engine(offset, optimizer_name):
    torch.manual_seed(1234)
    model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3))
    config = {
        "train_micro_batch_size_per_gpu": 3,
        "gradient_accumulation_steps": 2,
        "zero_optimization": {
            "stage": 3,
            "sub_group_size": 1
        },
    }
    if offset is not None:
        config["zero_optimization"]["optimizer_state_replica_rank_offset"] = offset
    if optimizer_name == "fused":
        config["optimizer"] = {"type": "Adam", "params": {"lr": 0.01}}
        config["fp16"] = {"enabled": True, "loss_scale": 128}
        optimizer = None
    elif optimizer_name == "none":
        optimizer = None
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, amsgrad=optimizer_name == "amsgrad")
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
    return engine


class TestZeroOptimizerStateReplica(DistributedTest):
    world_size = 3 if get_accelerator().device_name() == "cpu" else 2

    def test_completed_steps(self):
        if get_accelerator().device_name() not in ("cpu", "cuda"):
            pytest.skip("optimizer replicas require CPU/Gloo or CUDA/NCCL")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        offset = world_size - 1
        source = (rank - offset) % world_size
        optimizer_name = "adamw" if get_accelerator().device_name() == "cpu" else "fused"

        baseline = _make_engine(None, optimizer_name)
        _train(baseline, 0)
        _train(baseline, 1)
        baseline_first = _local_optimizer_state(baseline, include_padding=False)
        for micro_step in range(2, 6):
            _train(baseline, micro_step, overflow=micro_step == 3)
        baseline_final = _local_optimizer_state(baseline, include_padding=False)
        assert baseline.get_optimizer_state_replica(source) is None
        baseline.destroy()

        engine = _make_engine(offset, optimizer_name)
        assert engine.get_optimizer_state_replica(source) is None
        _train(engine, 0)
        assert engine.get_optimizer_state_replica(source) is None
        _train(engine, 1)
        assert engine.global_steps == 1
        first = engine.get_optimizer_state_replica(source, global_step=1)
        assert first is not None and first.source_rank == source and first.global_step == 1
        assert engine.get_optimizer_state_replica(rank) is None
        owner_state = _local_optimizer_state(engine)
        assert reduce_boolean_flags(_local_optimizer_state(engine, include_padding=False) == baseline_first)
        expected = [None] * world_size
        dist.all_gather_object(expected, owner_state)
        assert len(first.subgroups) > 1
        assert expected[source] != expected[rank]
        assert _replica_state(first) == expected[source]
        first_bits = expected[source]

        _train(engine, 2)
        assert engine.get_optimizer_state_replica(source) is first
        _train(engine, 3, overflow=True)
        assert engine.global_steps == 2 and engine.skipped_steps == 1
        assert engine.get_optimizer_state_replica(source, global_step=2) is None
        assert engine.get_optimizer_state_replica(source) is first

        _train(engine, 4)
        _train(engine, 5)
        assert engine.global_steps == 3
        final = engine.get_optimizer_state_replica(source, global_step=3)
        assert final is not None and final.source_rank == source and final.global_step == 3
        assert engine.get_optimizer_state_replica(source, global_step=1) is None
        owner_state = _local_optimizer_state(engine)
        assert reduce_boolean_flags(_local_optimizer_state(engine, include_padding=False) == baseline_final)
        dist.all_gather_object(expected, owner_state)
        assert _replica_state(final) == expected[source] != first_bits
        assert _replica_state(first) == first_bits
        dist.barrier()
        dist.destroy_process_group()
        assert not dist.is_initialized()
        assert _replica_state(engine.get_optimizer_state_replica(source, global_step=3)) == expected[source]

    @pytest.mark.world_size(2)
    def test_invalid_optimizer_or_offset(self):
        with pytest.raises(ValueError, match="less than world size"):
            _make_engine(dist.get_world_size(), "adamw")
        for optimizer in ("none", "sgd"):
            with pytest.raises(ValueError, match="requires Adam"):
                _make_engine(1, optimizer)
        with pytest.raises(ValueError, match="AMSGrad"):
            _make_engine(1, "amsgrad")
