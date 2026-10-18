# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, triton, tl

_CHUNK_SIZE = 4096

if _TRITON_AVAILABLE:

    @triton.jit
    def _chunk_gradient_statistics(pointer_table, lengths, chunk_tensors, chunk_offsets, partial_sum_squares,
                                   partial_invalid, BLOCK: tl.constexpr):
        chunk = tl.program_id(0)
        tensor_index = tl.load(chunk_tensors + chunk)
        base = tl.load(chunk_offsets + chunk)
        length = tl.load(lengths + tensor_index)
        gradient = tl.load(pointer_table + tensor_index).to(tl.pointer_type(tl.bfloat16))
        offsets = base + tl.arange(0, BLOCK)
        values = tl.load(gradient + offsets, mask=offsets < length, other=0.0).to(tl.float32)
        invalid = ((values != values) | (tl.abs(values) == float("inf"))).to(tl.int32)
        values = values.to(tl.float64)
        tl.store(partial_sum_squares + chunk, tl.sum(values * values, axis=0))
        tl.store(partial_invalid + chunk, tl.max(invalid, axis=0))

    @triton.jit
    def _tensor_gradient_statistics(partial_sum_squares, partial_invalid, chunk_starts, chunk_counts,
                                    tensor_sum_squares, tensor_invalid, REDUCE_BLOCK: tl.constexpr):
        tensor_index = tl.program_id(0)
        start = tl.load(chunk_starts + tensor_index)
        count = tl.load(chunk_counts + tensor_index)
        sum_squares = tl.zeros((), tl.float64)
        invalid = tl.zeros((), tl.int32)
        for base in tl.range(0, count, REDUCE_BLOCK):
            offsets = base + tl.arange(0, REDUCE_BLOCK)
            mask = offsets < count
            sum_squares += tl.sum(tl.load(partial_sum_squares + start + offsets, mask=mask, other=0.0), axis=0)
            invalid = tl.maximum(invalid, tl.max(tl.load(partial_invalid + start + offsets, mask=mask, other=0),
                                                 axis=0))
        tl.store(tensor_sum_squares + tensor_index, sum_squares)
        tl.store(tensor_invalid + tensor_index, invalid)


def run_fused_gradient_statistics(gradients,
                                  lengths,
                                  chunk_tensors,
                                  chunk_offsets,
                                  chunk_starts,
                                  chunk_counts,
                                  device,
                                  stable_descriptors=None):
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is unavailable")

    device = gradients[0].device if gradients else torch.device(device)
    if not gradients:
        return (torch.empty(0, dtype=torch.float64, device=device), torch.empty(0, dtype=torch.bool,
                                                                                device=device), (None, None))

    def make(values, dtype):
        return torch.tensor(values, dtype=dtype, device=device)

    pointers = make([gradient.data_ptr() for gradient in gradients], torch.int64)
    descriptor_key = (device, lengths)
    if stable_descriptors is None or stable_descriptors[0] != descriptor_key:
        stable_descriptors = (descriptor_key, make(lengths, torch.int64), make(chunk_tensors, torch.int32),
                              make(chunk_offsets, torch.int64), make(chunk_starts,
                                                                     torch.int64), make(chunk_counts, torch.int64))

    _, lengths_tensor, chunk_tensors_tensor, chunk_offsets_tensor, chunk_starts_tensor, chunk_counts_tensor = \
        stable_descriptors
    partial_sum_squares = torch.empty(len(chunk_tensors), dtype=torch.float64, device=device)
    partial_invalid = torch.empty(len(chunk_tensors), dtype=torch.int32, device=device)
    tensor_sum_squares = torch.empty(len(gradients), dtype=torch.float64, device=device)
    tensor_invalid = torch.empty(len(gradients), dtype=torch.int32, device=device)
    if chunk_tensors:
        _chunk_gradient_statistics[(len(chunk_tensors), )](pointers,
                                                           lengths_tensor,
                                                           chunk_tensors_tensor,
                                                           chunk_offsets_tensor,
                                                           partial_sum_squares,
                                                           partial_invalid,
                                                           BLOCK=_CHUNK_SIZE,
                                                           num_warps=8)
    _tensor_gradient_statistics[(len(gradients), )](partial_sum_squares,
                                                    partial_invalid,
                                                    chunk_starts_tensor,
                                                    chunk_counts_tensor,
                                                    tensor_sum_squares,
                                                    tensor_invalid,
                                                    REDUCE_BLOCK=256,
                                                    num_warps=4)
    keepalive = (pointers, stable_descriptors, partial_sum_squares, partial_invalid)
    return tensor_sum_squares, tensor_invalid.bool(), keepalive
