# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from unit.common import DistributedTest

# nccl_start_alignment_factor in DeepSpeedZeroOptimizer.
ALIGNMENT_FACTOR = 2


class EmptyTail(torch.nn.Module):
    """Holds the zero-sized parameter in a submodule so it is registered last.

    ``nn.Module.parameters()`` yields directly assigned parameters before
    submodule parameters, so a bare ``nn.Parameter`` attribute would be ordered
    *first* no matter where it is assigned. A leading zero-sized parameter starts
    at index 0, is registered in partition 0, and never triggers the failure this
    test covers.
    """

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(0))

    def forward(self):
        return self.weight.sum()


class ZeroSizedTailModel(torch.nn.Module):
    """Model whose last trainable parameter has zero elements.

    ZeRO-1/2 registers a parameter in ``param_to_partition_ids`` only when the
    parameter starts inside a partition range, or when a partition boundary falls
    strictly inside it. A zero-sized parameter has no extent, so a trailing one
    lands exactly on the end of the last partition and is registered nowhere,
    while its empty gradient still enters the reduction bucket. ``average_tensor``
    then performs an unguarded ``param_to_partition_ids[i][param_id]`` lookup and
    raises ``KeyError``.

    Two invariants keep this a regression test rather than a smoke test; both are
    asserted in the test body so a future edit fails loudly instead of silently
    passing with the fix reverted.

    1. The zero-sized parameter is one-dimensional. ``unflatten_dense_tensors``
       rebuilds a zero-numel entry as a fresh empty tensor, and before
       pytorch/pytorch#167976 (released in torch 2.10) that tensor was always
       1-D. ``_update_model_bit16_weights`` assigns it onto ``p.data``, so on
       torch < 2.10 a ``(0, N)`` parameter becomes ``(0,)`` during
       ``deepspeed.initialize()`` and the forward pass fails with a shape error
       before ZeRO reduction is reached. A 1-D zero-sized parameter round-trips
       unchanged on every supported torch.
    2. The flattened group must need no alignment padding, i.e. the element count
       must stay divisible by ``nccl_start_alignment_factor`` (2) times the world
       size. The two ``Linear(4, 4)`` modules supply 20 + 20 = 40 elements, which
       holds for world sizes 1, 2 and 4. With padding, the trailing parameter is
       registered after all and the failure does not reproduce.
    """

    def __init__(self, with_empty_param):
        super().__init__()
        self.dense = torch.nn.Linear(4, 4)
        self.tail = torch.nn.Linear(4, 4)
        self.empty = EmptyTail() if with_empty_param else None

    def forward(self, x):
        out = self.dense(x).pow(2).sum() + self.tail(x).pow(2).sum()
        if self.empty is not None:
            out = out + self.empty()
        return out


def _train_one_step(stage, with_empty_param):
    torch.manual_seed(1138)
    model = ZeroSizedTailModel(with_empty_param)
    params = list(model.parameters())
    if with_empty_param:
        assert params[-1].numel() == 0, "the zero-sized parameter must be the trailing one"
        assert sum(p.numel() for p in params) % (ALIGNMENT_FACTOR * dist.get_world_size()) == 0, \
            "the flattened group must need no alignment padding for this to reproduce"
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": stage
        },
    }
    engine, *_ = deepspeed.initialize(model=model,
                                      model_parameters=params,
                                      optimizer=torch.optim.AdamW(params, lr=1e-2),
                                      config=config)
    # Identical input on every rank, so the data-parallel average of the gradient
    # is the single-rank gradient and the step is deterministic across sizes.
    torch.manual_seed(2027)
    x = torch.randn(1, 4, device=engine.device, dtype=engine.module.dense.weight.dtype)
    engine.backward(engine(x))
    engine.step()
    kept = list(engine.module.dense.parameters()) + list(engine.module.tail.parameters())
    return [p.detach().float().cpu().clone() for p in kept]


@pytest.mark.parametrize("stage", [1, 2])
class TestZeroEmptyParam(DistributedTest):
    world_size = [1, 2]

    def test_backward_and_step(self, stage):
        torch.manual_seed(1138)
        reference = ZeroSizedTailModel(True)
        initial = [
            p.detach().float().clone() for p in list(reference.dense.parameters()) + list(reference.tail.parameters())
        ]

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
