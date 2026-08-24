# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""The fused local token engine against the eager reorder and weighted restore.

The eager implementations are the reference: the fused engine is only worth
having if it is indistinguishable from them, so every assertion here compares the
two directly rather than against hand-written expectations.
"""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.moe import autoep_fused_token_ops as fused_ops
from deepspeed.module_inject.auto_ep_layer import (
    combine_from_routed,
    permute_by_local_expert,
    unpermute_by_local_expert,
)


def _fused_engine_available():
    accelerator = get_accelerator()
    return (accelerator.is_available() and accelerator.device_name().startswith("cuda") and fused_ops.is_available())


pytestmark = pytest.mark.skipif(not _fused_engine_available(),
                                reason="the fused local token engine needs CUDA and Triton")

# Row counts per local expert, or per [source rank, local expert] where nested.
# They are the only thing that decides the reorder, so they carry every shape the
# engine has to survive: idle experts, one expert taking everything, and sources
# that contribute nothing to a given expert.
REORDER_CASES = {
    "balanced": [8, 8, 8, 8],
    "empty_experts": [0, 12, 0, 4],
    "extreme_skew": [40, 0, 0, 0],
    "per_source": [[3, 5], [7, 1]],
    "ragged_per_source": [[0, 9], [5, 0], [2, 2]],
}


def _device():
    return get_accelerator().current_device_name()


def _counts(case):
    return torch.tensor(REORDER_CASES[case], dtype=torch.int32, device=_device())


@pytest.mark.parametrize("case", sorted(REORDER_CASES))
@pytest.mark.parametrize("hidden", [64, 130])
def test_fused_reorder_places_the_same_rows_as_eager(case, hidden):
    counts = _counts(case)
    tokens = torch.randn(int(counts.sum()), hidden, device=_device(), dtype=torch.bfloat16)

    eager_rows, _permutation, eager_counts, _n_tokens = permute_by_local_expert(tokens, counts)
    fused_rows, context = fused_ops.fused_permute_by_local_expert(tokens, counts)

    # Pure data movement on both sides, so anything short of equality is a bug.
    assert torch.equal(fused_rows, eager_rows)
    assert torch.equal(context.aligned_counts, eager_counts)


def test_fused_reorder_handles_a_batch_no_expert_claimed():
    counts = torch.zeros(4, dtype=torch.int32, device=_device())
    tokens = torch.randn(0, 32, device=_device(), dtype=torch.bfloat16)

    eager_rows, _permutation, eager_counts, _n_tokens = permute_by_local_expert(tokens, counts)
    fused_rows, context = fused_ops.fused_permute_by_local_expert(tokens, counts)

    assert torch.equal(fused_rows, eager_rows)
    assert torch.equal(context.aligned_counts, eager_counts)
    assert not fused_rows.any()


@pytest.mark.parametrize("case", sorted(REORDER_CASES))
def test_fused_reorder_round_trip_matches_eager_including_gradients(case):
    hidden = 96
    counts = _counts(case)
    n_tokens = int(counts.sum())
    tokens = torch.randn(n_tokens, hidden, device=_device(), dtype=torch.bfloat16)
    upstream = torch.randn(n_tokens, hidden, device=_device(), dtype=torch.bfloat16)

    eager_tokens = tokens.clone().requires_grad_(True)
    eager_rows, permutation, _counts_out, n = permute_by_local_expert(eager_tokens, counts)
    # Scaling by a power of two keeps the comparison exact while still putting a
    # real op between the reorder and its inverse.
    eager_output = unpermute_by_local_expert(eager_rows * 2.0, permutation, n)

    fused_tokens = tokens.clone().requires_grad_(True)
    fused_rows, context = fused_ops.fused_permute_by_local_expert(fused_tokens, counts)
    fused_output = fused_ops.fused_unpermute_by_local_expert(fused_rows * 2.0, context)

    assert torch.equal(fused_output, eager_output)

    eager_output.backward(upstream)
    fused_output.backward(upstream)
    assert torch.equal(fused_tokens.grad, eager_tokens.grad)


@pytest.mark.parametrize("top_k", [2, 4, 6, 8])
@pytest.mark.parametrize("hidden", [128, 130])
@pytest.mark.parametrize("score_dtype", [torch.float32, torch.bfloat16])
def test_fused_weighted_restore_matches_eager_including_gradients(top_k, hidden, score_dtype):
    device = _device()
    num_tokens, num_experts = 24, 8
    generator = torch.Generator(device=device).manual_seed(20260824)

    selected_experts = torch.randint(0, num_experts, (num_tokens, top_k), device=device, generator=generator)
    token_indices_sorted = torch.argsort(selected_experts.view(-1), stable=True)
    # A restore that only had to undo the identity would not exercise anything.
    assert not torch.equal(token_indices_sorted, torch.arange(num_tokens * top_k, device=device))

    rows = torch.randn(num_tokens * top_k, hidden, device=device, dtype=torch.bfloat16, generator=generator)
    scores = torch.rand(num_tokens, top_k, device=device, dtype=score_dtype, generator=generator)
    upstream = torch.randn(1, num_tokens, hidden, device=device, dtype=torch.bfloat16, generator=generator)

    eager_rows = rows.clone().requires_grad_(True)
    eager_scores = scores.clone().requires_grad_(True)
    eager_output = combine_from_routed(
        eager_rows,
        top_scores=eager_scores,
        token_indices_sorted=token_indices_sorted,
        top_k=top_k,
        score_apply="post",
        combine_impl="weighted_sum",
        shape=(1, num_tokens, hidden),
    )

    fused_rows = rows.clone().requires_grad_(True)
    fused_scores = scores.clone().requires_grad_(True)
    fused_output = fused_ops.fused_weighted_restore(
        fused_rows,
        top_scores=fused_scores,
        token_indices_sorted=token_indices_sorted,
        top_k=top_k,
        shape=(1, num_tokens, hidden),
    )

    torch.testing.assert_close(fused_output, eager_output)

    eager_output.backward(upstream)
    fused_output.backward(upstream)

    torch.testing.assert_close(fused_rows.grad, eager_rows.grad)
    # The score gradient reduces over the hidden dimension, so the fused and eager
    # summation orders differ even though both accumulate in FP32. That shows up
    # in FP32 scores; a bfloat16 score rounds the difference away, and asking for
    # FP32 precision there would fail on a rounding boundary rather than on a bug.
    score_tolerance = {"rtol": 1e-4, "atol": 1e-5} if score_dtype == torch.float32 else {}
    torch.testing.assert_close(fused_scores.grad, eager_scores.grad, **score_tolerance)


def test_fused_weighted_restore_requires_one_row_per_assignment():
    device = _device()
    with pytest.raises(RuntimeError, match="one row per assignment"):
        fused_ops.fused_weighted_restore(
            torch.randn(10, 16, device=device, dtype=torch.bfloat16),
            top_scores=torch.rand(4, 2, device=device),
            token_indices_sorted=torch.arange(8, device=device),
            top_k=2,
            shape=(1, 4, 16),
        )


def test_fused_engine_names_what_it_cannot_run():
    device = _device()
    supported = torch.randn(8, 16, device=device, dtype=torch.bfloat16)
    fused_ops.assert_supported(supported, score_apply="post")

    with pytest.raises(RuntimeError, match="bfloat16 and float16"):
        fused_ops.assert_supported(torch.randn(8, 16, device=device, dtype=torch.float32), score_apply="post")

    with pytest.raises(RuntimeError, match='score_apply="post"'):
        fused_ops.assert_supported(supported, score_apply="pre")

    with pytest.raises(RuntimeError, match="CUDA kernels"):
        fused_ops.assert_supported(torch.randn(8, 16, dtype=torch.bfloat16), score_apply="post")
