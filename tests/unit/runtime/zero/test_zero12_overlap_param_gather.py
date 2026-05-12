# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

import pytest
import torch

import deepspeed
from deepspeed.runtime.zero import stage_1_and_2
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from unit.common import DistributedTest, preferred_dtype
from unit.simple_model import SimpleModel, random_dataloader


class _WaitHandle:

    def __init__(self):
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1


def _optimizer_shell():
    return object.__new__(DeepSpeedZeroOptimizer)


def test_zero12_overlap_param_block_map_uses_round_robin_flat_layout(monkeypatch):
    monkeypatch.setattr(stage_1_and_2.dist, "get_world_size", lambda group=None: 2)

    optimizer = _optimizer_shell()
    optimizer.nccl_start_alignment_factor = 1
    optimizer._zero12_overlap_param_gather_bucket_size = 8
    optimizer.real_dp_process_group = [object()]

    params = [
        torch.nn.Parameter(torch.empty(3)),
        torch.nn.Parameter(torch.empty(6)),
        torch.nn.Parameter(torch.empty(5)),
    ]
    optimizer.round_robin_bit16_groups = [params]
    optimizer.parallel_partitioned_bit16_groups = [[torch.empty(8), torch.empty(8)]]

    optimizer._zero12_overlap_build_param_block_map()

    assert optimizer._zero12_overlap_param_gather_block_sizes == [4]
    assert optimizer._zero12_overlap_param_gather_param_blocks[id(params[0])] == frozenset({(0, 0)})
    assert optimizer._zero12_overlap_param_gather_param_blocks[id(params[1])] == frozenset({(0, 0), (0, 1)})
    assert optimizer._zero12_overlap_param_gather_param_blocks[id(params[2])] == frozenset({(0, 0), (0, 1)})


def test_zero12_overlap_forward_hooks_wait_only_direct_parameter_blocks():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 2, bias=False),
    )

    optimizer = _optimizer_shell()
    optimizer._zero12_overlap_param_gather_param_blocks = {
        id(model[0].weight): frozenset({(0, 0)}),
        id(model[1].weight): frozenset({(0, 1), (0, 2)}),
    }
    optimizer._zero12_overlap_param_gather_module_blocks = {}
    optimizer._zero12_overlap_param_gather_hooks = []
    waited_blocks = []
    optimizer._zero12_overlap_wait_blocks = lambda blocks: waited_blocks.append(frozenset(blocks))

    optimizer._zero12_overlap_register_module_hooks(model)
    try:
        model[0](torch.ones(1, 4))
        model[1](torch.ones(1, 4))
    finally:
        for hook in optimizer._zero12_overlap_param_gather_hooks:
            hook.remove()

    assert waited_blocks == [frozenset({(0, 0)}), frozenset({(0, 1), (0, 2)})]


def test_zero12_overlap_wait_blocks_leaves_unrequested_work_pending():
    optimizer = _optimizer_shell()
    handles = [_WaitHandle(), _WaitHandle(), _WaitHandle()]
    optimizer._zero12_overlap_param_gather_pending = {
        (0, 0): (handles[0], []),
        (0, 1): (handles[1], []),
        (0, 2): (handles[2], []),
    }

    optimizer._zero12_overlap_wait_blocks({(0, 1)})

    assert handles[0].wait_count == 0
    assert handles[1].wait_count == 1
    assert handles[2].wait_count == 0
    assert set(optimizer._zero12_overlap_param_gather_pending) == {(0, 0), (0, 2)}


@pytest.mark.parametrize("zero_stage", [1, 2])
class TestZero12OverlapParamGatherSmoke(DistributedTest):
    world_size = 2

    def test_runtime_smoke(self, zero_stage):
        dtype = preferred_dtype()
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("ZeRO-1/2 overlapped parameter refresh requires fp16 or bf16 training")

        old_enabled = os.environ.get("DEEPSPEED_ZERO12_OVERLAP_PARAM_GATHER")
        old_bucket = os.environ.get("DEEPSPEED_ZERO12_OVERLAP_PARAM_GATHER_BUCKET_SIZE")
        os.environ["DEEPSPEED_ZERO12_OVERLAP_PARAM_GATHER"] = "1"
        os.environ["DEEPSPEED_ZERO12_OVERLAP_PARAM_GATHER_BUCKET_SIZE"] = "16"

        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": zero_stage,
                "allgather_bucket_size": 16,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3,
                    "torch_adam": True,
                },
            },
        }
        if dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}
        else:
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}

        engine = None
        try:
            model = SimpleModel(hidden_dim=4, nlayers=2)
            engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                                   model=model,
                                                   model_parameters=model.parameters())
            assert engine.optimizer._zero12_overlap_param_gather_enabled

            data_loader = random_dataloader(model=engine,
                                            total_samples=2,
                                            hidden_dim=4,
                                            device=engine.device,
                                            dtype=dtype)
            batch = next(iter(data_loader))
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()
            engine.optimizer._zero12_overlap_wait_all_pending("test assertion")
            assert not engine.optimizer._zero12_overlap_param_gather_pending
        finally:
            if engine is not None:
                engine.destroy()
            if old_enabled is None:
                os.environ.pop("DEEPSPEED_ZERO12_OVERLAP_PARAM_GATHER", None)
            else:
                os.environ["DEEPSPEED_ZERO12_OVERLAP_PARAM_GATHER"] = old_enabled
            if old_bucket is None:
                os.environ.pop("DEEPSPEED_ZERO12_OVERLAP_PARAM_GATHER_BUCKET_SIZE", None)
            else:
                os.environ["DEEPSPEED_ZERO12_OVERLAP_PARAM_GATHER_BUCKET_SIZE"] = old_bucket
