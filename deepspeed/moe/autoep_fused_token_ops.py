# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Fused weighted token restoration for AutoEP.

After the combine all-to-all, the eager path returns one row per routed
assignment and turns it back into one row per token in general-purpose steps: it
scatters the rows into a zero-filled ``[tokens * top_k, hidden]`` buffer, views
that as ``[tokens, top_k, hidden]``, widens it to FP32 to apply routing weights,
and reduces over top-k. The FP32 intermediate alone is 64 MiB at the canonical
shape, and every step of that sequence costs a full pass over the routed
activations, in every MoE layer, on every step.

This module does the same arithmetic in one pass. Each program owns one token
and one slice of the hidden dimension, walks its top-k rows in registers,
accumulates in FP32 and writes the token's output once, so neither the scattered
assignment buffer nor the FP32 intermediate is ever allocated.

Only the reduction is replaced. The collectives, the router, the grouped GEMM
and the expert-major reorder are all untouched, so a measured difference belongs
to the reduction alone.
"""

from __future__ import annotations

import torch

_IS_ROCM_PYTORCH = getattr(torch.version, "hip", None) is not None

if _IS_ROCM_PYTORCH:
    _TRITON_AVAILABLE = False
else:
    try:
        import triton
        import triton.language as tl

        _TRITON_AVAILABLE = True
    except ImportError:
        _TRITON_AVAILABLE = False

# The grouped GEMM produces the rows this consumes, so the supported dtypes are
# the ones it is built for rather than a silent widening.
SUPPORTED_ROW_DTYPES = (torch.bfloat16, torch.float16)

_MAX_BLOCK_HIDDEN = 512
_INVERT_INDEX_BLOCK = 256
# The kernels hold a [slots, BLOCK_H] FP32 block live, so the hidden tile shrinks
# as top-k grows to keep that block in registers rather than spilling.
_MAX_BLOCK_ELEMENTS = 2048

if _TRITON_AVAILABLE:

    @triton.jit
    def _invert_index_kernel(
        index_ptr,
        inverse_ptr,
        num_indices,
        num_inverse_rows,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        in_range = offsets < num_indices

        targets = tl.load(index_ptr + offsets, mask=in_range, other=-1).to(tl.int64)
        writable = in_range & (targets >= 0) & (targets < num_inverse_rows)
        tl.store(inverse_ptr + tl.where(writable, targets, 0), offsets.to(tl.int32), mask=writable)

    @triton.jit
    def _weighted_restore_forward_kernel(
        rows_ptr,
        inverse_ptr,
        scores_ptr,
        out_ptr,
        hidden,
        rows_stride,
        scores_stride,
        out_stride,
        TOP_K: tl.constexpr,
        K_PADDED: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token = tl.program_id(0)
        hidden_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        hidden_mask = hidden_offsets < hidden

        slots = tl.arange(0, K_PADDED)
        slot_mask = slots < TOP_K

        source_rows = tl.load(inverse_ptr + token * TOP_K + slots, mask=slot_mask, other=-1).to(tl.int64)
        row_valid = slot_mask & (source_rows >= 0)
        safe_rows = tl.where(row_valid, source_rows, 0)

        scores = tl.load(scores_ptr + token * scores_stride + slots, mask=slot_mask, other=0.0).to(tl.float32)

        block_mask = row_valid[:, None] & hidden_mask[None, :]
        values = tl.load(
            rows_ptr + safe_rows[:, None] * rows_stride + hidden_offsets[None, :],
            mask=block_mask,
            other=0.0,
        ).to(tl.float32)

        # FP32 product and reduction with a single cast on the way out, matching
        # the dtype discipline of the eager weighted sum.
        weighted = tl.sum(values * scores[:, None], axis=0)
        tl.store(
            out_ptr + token * out_stride + hidden_offsets,
            weighted.to(out_ptr.dtype.element_ty),
            mask=hidden_mask,
        )

    @triton.jit
    def _weighted_restore_backward_kernel(
        grad_out_ptr,
        rows_ptr,
        inverse_ptr,
        scores_ptr,
        grad_rows_ptr,
        grad_scores_ptr,
        hidden,
        grad_out_stride,
        rows_stride,
        scores_stride,
        grad_rows_stride,
        grad_scores_stride,
        TOP_K: tl.constexpr,
        K_PADDED: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token = tl.program_id(0)

        slots = tl.arange(0, K_PADDED)
        slot_mask = slots < TOP_K

        source_rows = tl.load(inverse_ptr + token * TOP_K + slots, mask=slot_mask, other=-1).to(tl.int64)
        row_valid = slot_mask & (source_rows >= 0)
        safe_rows = tl.where(row_valid, source_rows, 0)
        scores = tl.load(scores_ptr + token * scores_stride + slots, mask=slot_mask, other=0.0).to(tl.float32)

        grad_rows_dtype = grad_rows_ptr.dtype.element_ty
        # One token per program, so the score gradient reduces over the hidden
        # dimension in registers. Splitting that dimension across programs and
        # reducing the partials afterwards measured slower at this shape: the
        # extra pass costs more than the added parallelism returns.
        score_partials = tl.zeros([K_PADDED, BLOCK_H], dtype=tl.float32)

        for hidden_start in range(0, hidden, BLOCK_H):
            hidden_offsets = hidden_start + tl.arange(0, BLOCK_H)
            hidden_mask = hidden_offsets < hidden
            block_mask = row_valid[:, None] & hidden_mask[None, :]

            upstream = tl.load(
                grad_out_ptr + token * grad_out_stride + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)

            values = tl.load(
                rows_ptr + safe_rows[:, None] * rows_stride + hidden_offsets[None, :],
                mask=block_mask,
                other=0.0,
            ).to(tl.float32)
            score_partials += values * upstream[None, :]

            tl.store(
                grad_rows_ptr + safe_rows[:, None] * grad_rows_stride + hidden_offsets[None, :],
                (upstream[None, :] * scores[:, None]).to(grad_rows_dtype),
                mask=block_mask,
            )

        grad_scores = tl.sum(score_partials, axis=1)
        tl.store(
            grad_scores_ptr + token * grad_scores_stride + slots,
            grad_scores.to(grad_scores_ptr.dtype.element_ty),
            mask=slot_mask,
        )


def is_available() -> bool:
    """Whether this build can run the fused weighted restore at all."""
    return _TRITON_AVAILABLE


def assert_supported(rows: torch.Tensor, *, score_apply: str) -> None:
    """Reject configurations the fused restore does not implement.

    Checked before any collective runs: a rank that raised while its peers
    proceeded would turn a clear error into a hang.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError('combine_impl="fused_weighted_sum" needs Triton, which is not installed in this '
                           "environment. Install Triton, or leave combine_impl unset.")
    if rows.device.type != "cuda":
        raise RuntimeError('combine_impl="fused_weighted_sum" runs CUDA kernels but this layer is on device '
                           f'"{rows.device.type}". Leave combine_impl unset to run here.')
    if rows.dtype not in SUPPORTED_ROW_DTYPES:
        raise RuntimeError('combine_impl="fused_weighted_sum" supports bfloat16 and float16 rows, got '
                           f"{rows.dtype}. Leave combine_impl unset, or train in bf16/fp16.")
    if score_apply != "post":
        raise RuntimeError('combine_impl="fused_weighted_sum" folds the routing weight into the top-k reduction, '
                           f'which only exists for score_apply="post", but this layer resolved '
                           f'score_apply="{score_apply}". Leave combine_impl unset.')


def _block_hidden(hidden: int, slots: int) -> int:
    """Pick a power-of-two hidden tile that fits alongside ``slots`` rows of FP32.

    The floor keeps the budget honest for top-k values far wider than any real
    router, so the tile shrinks rather than overrunning the element budget.
    """
    budget = max(16, _MAX_BLOCK_ELEMENTS // slots)
    return min(_MAX_BLOCK_HIDDEN, budget, max(16, triton.next_power_of_2(hidden)))


def _padded_top_k(top_k: int) -> int:
    """Round top-k up to a power of two, which ``tl.arange`` requires."""
    return max(2, triton.next_power_of_2(top_k))


def _invert_index(index: torch.Tensor, num_inverse_rows: int) -> torch.Tensor:
    """Invert a row permutation, leaving -1 wherever no slot claimed a row."""
    inverse = torch.full((num_inverse_rows, ), -1, dtype=torch.int32, device=index.device)
    num_indices = index.numel()
    if num_indices == 0 or num_inverse_rows == 0:
        return inverse

    grid = (triton.cdiv(num_indices, _INVERT_INDEX_BLOCK), )
    _invert_index_kernel[grid](
        index.contiguous(),
        inverse,
        num_indices,
        num_inverse_rows,
        BLOCK=_INVERT_INDEX_BLOCK,
    )
    return inverse


def _differentiable_backward(grad_output, combined_rows, top_scores, inverse, top_k):
    """Build the rare higher-order backward with regular PyTorch operations."""
    n_tokens, hidden = top_scores.shape[0], combined_rows.shape[-1]
    valid = inverse >= 0
    safe_inverse = inverse.clamp_min(0).to(torch.int64)
    gathered_rows = combined_rows.index_select(0, safe_inverse).reshape(n_tokens, top_k, hidden)

    grad_by_assignment = (grad_output[:, None, :] * top_scores[:, :, None]).to(combined_rows.dtype).reshape(-1, hidden)
    grad_rows = torch.zeros_like(combined_rows)
    grad_rows = grad_rows.index_copy(0, safe_inverse[valid], grad_by_assignment[valid])

    grad_scores = (gathered_rows.float() * grad_output.float()[:, None, :]).sum(dim=-1)
    grad_scores = torch.where(valid.reshape(n_tokens, top_k), grad_scores, 0.0).to(top_scores.dtype)
    return grad_rows, grad_scores


class _FusedWeightedRestore(torch.autograd.Function):
    """Weight rows by their routing score and reduce over top-k in one pass."""

    @staticmethod
    def forward(ctx, combined_rows, top_scores, inverse, top_k):
        combined_rows = combined_rows.contiguous()
        n_tokens, hidden = top_scores.shape[0], combined_rows.shape[-1]
        output = torch.empty((n_tokens, hidden), dtype=combined_rows.dtype, device=combined_rows.device)

        ctx.save_for_backward(combined_rows, top_scores, inverse)
        ctx.top_k = top_k
        if n_tokens == 0 or hidden == 0:
            return output

        k_padded = _padded_top_k(top_k)
        block_hidden = _block_hidden(hidden, slots=k_padded)
        grid = (n_tokens, triton.cdiv(hidden, block_hidden))
        _weighted_restore_forward_kernel[grid](
            combined_rows,
            inverse,
            top_scores,
            output,
            hidden,
            combined_rows.stride(0),
            top_scores.stride(0),
            output.stride(0),
            TOP_K=top_k,
            K_PADDED=k_padded,
            BLOCK_H=block_hidden,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        combined_rows, top_scores, inverse = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        if torch.is_grad_enabled():
            grad_rows, grad_scores = _differentiable_backward(grad_output, combined_rows, top_scores, inverse,
                                                              ctx.top_k)
            return grad_rows, grad_scores, None, None

        # AutoEP supplies an exact permutation, so every row is written once.
        # Zero initialization also keeps malformed direct calls deterministic
        # when an invalid or duplicate assignment leaves an inverse slot empty.
        grad_rows = torch.zeros_like(combined_rows)
        grad_scores = torch.empty_like(top_scores)

        n_tokens, hidden = top_scores.shape[0], combined_rows.shape[-1]
        if n_tokens == 0 or hidden == 0:
            # Nothing is reduced, so the score gradient is zero rather than
            # whatever an uninitialized buffer happened to hold.
            return grad_rows, torch.zeros_like(top_scores), None, None

        k_padded = _padded_top_k(ctx.top_k)
        _weighted_restore_backward_kernel[(n_tokens, )](
            grad_output,
            combined_rows,
            inverse,
            top_scores,
            grad_rows,
            grad_scores,
            hidden,
            grad_output.stride(0),
            combined_rows.stride(0),
            top_scores.stride(0),
            grad_rows.stride(0),
            grad_scores.stride(0),
            TOP_K=ctx.top_k,
            K_PADDED=k_padded,
            BLOCK_H=_block_hidden(hidden, slots=k_padded),
        )
        return grad_rows, grad_scores, None, None


def fused_weighted_restore(
    combined_rows: torch.Tensor,
    top_scores: torch.Tensor,
    token_indices_sorted: torch.Tensor,
    top_k: int,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """Weight combined rows by their routing scores and reduce over top-k.

    Fused counterpart of ``combine_from_routed`` for ``score_apply="post"``. It
    goes straight from ``[T * K, H]`` to ``[B, S, H]``, so neither the scattered
    assignment buffer nor the ``[T, K, H]`` FP32 intermediate is allocated.
    """
    bsz, seqlen, hidden = shape
    if top_k <= 0:
        raise RuntimeError(f"fused weighted restore expects top_k > 0, got {top_k}.")
    if bsz < 0 or seqlen < 0 or hidden < 0:
        raise RuntimeError(f"fused weighted restore expects non-negative output dimensions, got {shape}.")
    if combined_rows.ndim != 2:
        raise RuntimeError(f"fused weighted restore expects combined_rows to be 2D, got shape "
                           f"{tuple(combined_rows.shape)}.")
    if combined_rows.shape[1] != hidden:
        raise RuntimeError(f"fused weighted restore output hidden size is {hidden}, but combined rows have hidden "
                           f"size {combined_rows.shape[1]}.")

    n_tokens = bsz * seqlen
    expected_rows = n_tokens * top_k
    if combined_rows.shape[0] != expected_rows:
        raise RuntimeError(f"fused weighted restore expects one row per assignment: {expected_rows} rows for "
                           f"{n_tokens} tokens at top_k={top_k}, got {combined_rows.shape[0]}.")
    if tuple(top_scores.shape) != (n_tokens, top_k):
        raise RuntimeError(f"fused weighted restore expects top_scores shape {(n_tokens, top_k)}, got "
                           f"{tuple(top_scores.shape)}.")
    if token_indices_sorted.ndim != 1 or token_indices_sorted.numel() != expected_rows:
        raise RuntimeError(f"fused weighted restore expects token_indices_sorted to contain {expected_rows} "
                           f"assignments, got shape {tuple(token_indices_sorted.shape)}.")
    if token_indices_sorted.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("fused weighted restore expects token_indices_sorted to use int32 or int64 indices, got "
                           f"{token_indices_sorted.dtype}.")
    if not torch.is_floating_point(top_scores):
        raise RuntimeError(f"fused weighted restore expects floating-point top_scores, got {top_scores.dtype}.")
    if combined_rows.device != top_scores.device or combined_rows.device != token_indices_sorted.device:
        raise RuntimeError("fused weighted restore expects rows, scores, and indices on the same device, got "
                           f"{combined_rows.device}, {top_scores.device}, and {token_indices_sorted.device}.")

    inverse = _invert_index(token_indices_sorted, expected_rows)
    output = _FusedWeightedRestore.apply(combined_rows, top_scores.contiguous(), inverse, top_k)
    return output.reshape(bsz, seqlen, hidden)
