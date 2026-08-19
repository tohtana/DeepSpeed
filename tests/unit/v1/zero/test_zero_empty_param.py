# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
from unit.common import DistributedTest


class ZeroSizedTailModel(torch.nn.Module):
    """Model whose last trainable parameter has zero elements.

    ``ZeRO-1/2`` registers a parameter in ``param_to_partition_ids`` only when the
    parameter starts inside a partition range or a partition boundary falls
    strictly inside it. A zero-sized parameter has no extent, so a trailing one
    lands exactly on the end of the last partition and is registered nowhere,
    while its empty gradient still enters the reduction bucket.

    That only happens when the flattened group needs no alignment padding, i.e.
    when the total element count is divisible by
    ``nccl_start_alignment_factor (2) * world_size``. ``Linear(4, 4)`` supplies
    16 + 4 = 20 elements, which satisfies that for the world sizes used here.
    Keep that invariant if this model is ever resized, otherwise the padding
    hides the failure and this test stops being a regression test.
    """

    def __init__(self, with_empty_param):
        super().__init__()
        self.dense = torch.nn.Linear(4, 4)
        self.empty = torch.nn.Linear(4, 0, bias=False) if with_empty_param else None

    def forward(self, x):
        out = self.dense(x).pow(2).sum()
        if self.empty is not None:
            out = out + self.empty(x).sum()
        return out


def _train_one_step(stage, with_empty_param):
    torch.manual_seed(1138)
    model = ZeroSizedTailModel(with_empty_param)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": stage
        },
    }
    engine, *_ = deepspeed.initialize(model=model,
                                      model_parameters=model.parameters(),
                                      optimizer=optimizer,
                                      config=config)
    # Identical input on every rank, so the data-parallel average of the
    # gradient is the single-rank gradient and the step is deterministic.
    torch.manual_seed(2027)
    x = torch.randn(1, 4, device=engine.device, dtype=engine.module.dense.weight.dtype)
    engine.backward(engine(x))
    engine.step()
    return [p.detach().float().cpu().clone() for p in engine.module.dense.parameters()]


@pytest.mark.parametrize("stage", [1, 2])
class TestZeroEmptyParam(DistributedTest):
    world_size = [1, 2]

    def test_backward_and_step(self, stage):
        torch.manual_seed(1138)
        initial = [p.detach().float().clone() for p in ZeroSizedTailModel(True).dense.parameters()]

        with_empty = _train_one_step(stage, with_empty_param=True)

        # The step must actually have run, not silently no-opped.
        for before, after in zip(initial, with_empty):
            assert torch.isfinite(after).all()
            assert not torch.equal(before, after)

        # A zero-element parameter contributes nothing to the flattened group, so
        # its presence must not perturb the reduction of the other parameters.
        without_empty = _train_one_step(stage, with_empty_param=False)
        for lhs, rhs in zip(with_empty, without_empty):
            assert torch.equal(lhs, rhs)
