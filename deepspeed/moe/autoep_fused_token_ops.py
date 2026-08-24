# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Fused GPU-local token movement for AutoEP.

The eager path moves routed rows around the two expert all-to-alls with
general-purpose tensor ops: it materializes a padded copy of the token matrix,
gathers through an advanced index, scatters expert outputs into a zero-filled
buffer, and builds a ``[T, K, H]`` FP32 intermediate only to apply routing
weights and reduce over top-k.

This module replaces that sequence with Triton kernels that touch each row once.
It deliberately leaves the collectives, the router and the grouped GEMM alone, so
that a measured difference is attributable to the local token engine.

The reorder is expressed entirely as row gathers. Writing the four passes out
shows why only one kernel is needed, where ``perm`` maps an expert-major slot to
its source row and ``inv`` is its inverse:

    permute   forward   out[i]  = tokens[perm[i]]      gather by perm
    permute   backward  dtok[j] = dout[inv[j]]         gather by inv
    unpermute forward   out[j]  = expert_out[inv[j]]   gather by inv
    unpermute backward  dexp[i] = dout[perm[i]]        gather by perm

``perm`` is injective on real rows, so no pass needs atomics, and carrying
``inv`` (4 bytes per row) removes both the padded input copy and the zero-filled
scatter buffer (``2 * H`` bytes per row) that the eager path allocates.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# The grouped GEMM consumes the reordered rows, so the fused path supports the
# dtypes it is built for rather than silently widening them.
SUPPORTED_ROW_DTYPES = (torch.bfloat16, torch.float16)

_MAX_BLOCK_HIDDEN = 512
_INVERT_INDEX_BLOCK = 256
# The restore kernels hold a [slots, BLOCK_H] FP32 block live, so the hidden tile
# shrinks as top-k grows to keep that block in registers instead of spilling.
_MAX_BLOCK_ELEMENTS = 2048

if _TRITON_AVAILABLE:

    @triton.jit
    def _gather_rows_kernel(
        source_ptr,
        index_ptr,
        out_ptr,
        hidden,
        source_stride,
        out_stride,
        BLOCK_H: tl.constexpr,
    ):
        out_row = tl.program_id(0)
        hidden_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        hidden_mask = hidden_offsets < hidden

        source_row = tl.load(index_ptr + out_row).to(tl.int64)
        # A negative index marks an alignment-padding slot. The eager path spelled
        # that as an appended zero row; here it is a masked-off load.
        row_valid = source_row >= 0
        safe_row = tl.where(row_valid, source_row, 0)

        values = tl.load(
            source_ptr + safe_row * source_stride + hidden_offsets,
            mask=hidden_mask & row_valid,
            other=0.0,
        )
        tl.store(out_ptr + out_row * out_stride + hidden_offsets, values, mask=hidden_mask)

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
        # dimension in registers instead of through a cross-program atomic.
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


class FusedReorderContext(NamedTuple):
    """Index metadata shared by the reorder forward and backward passes."""

    permutation: torch.Tensor  # [N_padded] int32; -1 marks an alignment-padding slot
    inverse: torch.Tensor  # [n_tokens] int32; -1 marks a row no slot claimed
    aligned_counts: torch.Tensor  # [E_local] int32 row counts for the grouped GEMM
    n_tokens: int


def is_available() -> bool:
    """Whether this build can run the fused local token engine at all."""
    return _TRITON_AVAILABLE


def assert_supported(rows: torch.Tensor, *, score_apply: str) -> None:
    """Reject configurations the fused path does not implement.

    Checked before any collective runs: a rank that raised while its peers
    proceeded would turn a clear error into a hang.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError('expert_parallel.local_token_backend="fused" needs Triton, which is not installed in '
                           'this environment. Install Triton, or set local_token_backend to "eager".')
    if rows.device.type != "cuda":
        raise RuntimeError('expert_parallel.local_token_backend="fused" runs CUDA kernels but this layer is on '
                           f'device "{rows.device.type}". Set local_token_backend to "eager" to run here.')
    if rows.dtype not in SUPPORTED_ROW_DTYPES:
        raise RuntimeError('expert_parallel.local_token_backend="fused" supports bfloat16 and float16 rows, got '
                           f'{rows.dtype}. Set local_token_backend to "eager", or train in bf16/fp16.')
    if score_apply != "post":
        raise RuntimeError('expert_parallel.local_token_backend="fused" implements the post-expert weighted '
                           f'restore, but this layer resolved score_apply="{score_apply}". Set local_token_backend '
                           'to "eager".')


def _block_hidden(hidden: int, slots: int = 1) -> int:
    """Pick a power-of-two hidden tile that fits alongside ``slots`` rows of FP32.

    The floor keeps the budget honest for top-k values far wider than any real
    router, so the tile shrinks rather than overrunning the element budget.
    """
    budget = max(16, _MAX_BLOCK_ELEMENTS // slots)
    return min(_MAX_BLOCK_HIDDEN, budget, max(16, triton.next_power_of_2(hidden)))


def _padded_top_k(top_k: int) -> int:
    """Round top-k up to a power of two, which ``tl.arange`` requires."""
    return max(2, triton.next_power_of_2(top_k))


def _gather_rows(source: torch.Tensor, index: torch.Tensor, num_out_rows: int) -> torch.Tensor:
    """Gather the ``source`` rows named by ``index``, reading a negative index as zero."""
    hidden = source.shape[-1]
    out = torch.empty((num_out_rows, hidden), dtype=source.dtype, device=source.device)
    if num_out_rows == 0 or hidden == 0:
        return out

    source = source.contiguous()
    block_hidden = _block_hidden(hidden)
    grid = (num_out_rows, triton.cdiv(hidden, block_hidden))
    _gather_rows_kernel[grid](
        source,
        index,
        out,
        hidden,
        source.stride(0),
        out.stride(0),
        BLOCK_H=block_hidden,
    )
    return out


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


class _FusedReorder(torch.autograd.Function):
    """Gather routed rows into expert-major, alignment-padded order."""

    @staticmethod
    def forward(ctx, tokens, permutation, inverse, n_padded):
        ctx.save_for_backward(inverse)
        ctx.n_tokens = tokens.shape[0]
        return _gather_rows(tokens, permutation, n_padded)

    @staticmethod
    def backward(ctx, grad_out):
        inverse, = ctx.saved_tensors
        return _gather_rows(grad_out.contiguous(), inverse, ctx.n_tokens), None, None, None


class _FusedInverseReorder(torch.autograd.Function):
    """Restore source-major order and drop the alignment padding."""

    @staticmethod
    def forward(ctx, expert_output, permutation, inverse):
        ctx.save_for_backward(permutation)
        ctx.n_padded = expert_output.shape[0]
        return _gather_rows(expert_output, inverse, inverse.numel())

    @staticmethod
    def backward(ctx, grad_out):
        permutation, = ctx.saved_tensors
        return _gather_rows(grad_out.contiguous(), permutation, ctx.n_padded), None, None


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

        # Every row is claimed by exactly one slot, which fused_weighted_restore
        # checks by shape before building the inverse, so both gradients are
        # written in full and neither buffer needs pre-zeroing.
        grad_rows = torch.empty_like(combined_rows)
        grad_scores = torch.empty_like(top_scores)

        n_tokens, hidden = top_scores.shape[0], combined_rows.shape[-1]
        if n_tokens == 0 or hidden == 0:
            # Nothing is reduced, so the score gradient is zero rather than
            # whatever the uninitialized buffer happened to hold.
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


def fused_permute_by_local_expert(
    tokens: torch.Tensor,
    local_counts: torch.Tensor,
) -> tuple[torch.Tensor, FusedReorderContext]:
    """Reorder routed rows into expert-contiguous, alignment-padded order.

    Fused counterpart of ``permute_by_local_expert``. It shares that function's
    index generation and produces the same rows, without materializing the padded
    copy of ``tokens`` that the eager advanced index needs.
    """
    from deepspeed.moe.ep_kernels import generate_local_expert_permute_indices

    permutation, aligned_counts = generate_local_expert_permute_indices(
        n_tokens=tokens.shape[0],
        local_counts=local_counts,
        device=tokens.device,
    )
    inverse = _invert_index(permutation, tokens.shape[0])
    context = FusedReorderContext(
        permutation=permutation,
        inverse=inverse,
        aligned_counts=aligned_counts,
        n_tokens=tokens.shape[0],
    )
    permuted = _FusedReorder.apply(tokens, permutation, inverse, permutation.numel())
    return permuted, context


def fused_unpermute_by_local_expert(
    expert_output: torch.Tensor,
    context: FusedReorderContext,
) -> torch.Tensor:
    """Reverse :func:`fused_permute_by_local_expert` and strip the padding."""
    return _FusedInverseReorder.apply(expert_output.contiguous(), context.permutation, context.inverse)


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
    n_tokens = bsz * seqlen
    expected_rows = n_tokens * top_k
    if combined_rows.shape[0] != expected_rows:
        raise RuntimeError(f"fused weighted restore expects one row per assignment: {expected_rows} rows for "
                           f"{n_tokens} tokens at top_k={top_k}, got {combined_rows.shape[0]}.")

    inverse = _invert_index(token_indices_sorted, expected_rows)
    output = _FusedWeightedRestore.apply(combined_rows, top_scores.contiguous(), inverse, top_k)
    return output.reshape(bsz, seqlen, hidden)
