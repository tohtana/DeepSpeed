# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass, replace

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, triton, tl
from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.runtime.utils import is_model_parallel_parameter

_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class GradientStatisticsPlan:
    gradients: tuple
    lengths: tuple
    chunk_tensors: tuple
    chunk_offsets: tuple
    chunk_starts: tuple
    chunk_counts: tuple
    group_selected: tuple
    group_keys: tuple


def build_gradient_statistics_plan(gradient_groups,
                                   parameter_groups,
                                   model_parallel_rank,
                                   chunk_size=_CHUNK_SIZE,
                                   cached_plan=None):
    """Build tensor and chunk descriptors while preserving ZeRO's norm filters."""
    if len(gradient_groups) != len(parameter_groups):
        raise ValueError("gradient and parameter group counts differ")

    gradients = []
    lengths = []
    group_selected = []
    group_keys = []
    for gradient_group, parameter_group in zip(gradient_groups, parameter_groups):
        if gradient_group is None:
            raise ValueError("averaged gradient group is not populated")

        group_keys.append((id(gradient_group), id(parameter_group)))
        positions = []
        for gradient in gradient_group:
            if gradient is None:
                raise ValueError("averaged gradient is not populated")
            positions.append(len(gradients))
            gradients.append(gradient)
            lengths.append(gradient.numel())

        selected = []
        nonpadding = (position for position, gradient in zip(positions, gradient_group)
                      if not getattr(gradient, "_zero_padding", False))
        for position, parameter in zip(nonpadding, parameter_group):
            if getattr(parameter, PIPE_REPLICATED, False):
                continue
            if is_model_parallel_parameter(parameter) or model_parallel_rank == 0:
                selected.append(position)
        group_selected.append(tuple(selected))

    lengths = tuple(lengths)
    if cached_plan is not None and cached_plan.lengths == lengths:
        chunk_tensors = cached_plan.chunk_tensors
        chunk_offsets = cached_plan.chunk_offsets
        chunk_starts = cached_plan.chunk_starts
        chunk_counts = cached_plan.chunk_counts
    else:
        chunk_tensors = []
        chunk_offsets = []
        chunk_starts = []
        chunk_counts = []
        for tensor_index, length in enumerate(lengths):
            chunk_starts.append(len(chunk_tensors))
            chunk_counts.append((length + chunk_size - 1) // chunk_size)
            for offset in range(0, length, chunk_size):
                chunk_tensors.append(tensor_index)
                chunk_offsets.append(offset)
        chunk_tensors = tuple(chunk_tensors)
        chunk_offsets = tuple(chunk_offsets)
        chunk_starts = tuple(chunk_starts)
        chunk_counts = tuple(chunk_counts)

    return GradientStatisticsPlan(tuple(gradients), lengths, chunk_tensors, chunk_offsets, chunk_starts, chunk_counts,
                                  tuple(group_selected), tuple(group_keys))


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


def _preflight_plan(plan):
    if not plan.gradients:
        return None

    device = plan.gradients[0].device
    for gradient in plan.gradients:
        if gradient.device != device or gradient.device.type != "cuda":
            return "gradients must be on one CUDA device"
        if gradient.dtype != torch.bfloat16:
            return "gradients must use BF16"
        if gradient.layout != torch.strided or not gradient.is_contiguous():
            return "gradients must be dense and contiguous"
    return None


def _run_fused_gradient_statistics(plan, stable_descriptors=None):
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is unavailable")
    if not plan.gradients:
        device = torch.device(get_accelerator().current_device_name())
        return (torch.empty(0, dtype=torch.float64, device=device), torch.empty(0, dtype=torch.bool,
                                                                                device=device), (None, None))

    device = plan.gradients[0].device

    def make(values, dtype):
        return torch.tensor(values, dtype=dtype, device=device)

    pointers = make([gradient.data_ptr() for gradient in plan.gradients], torch.int64)
    descriptor_key = (device, plan.lengths)
    if stable_descriptors is None or stable_descriptors[0] != descriptor_key:
        stable_descriptors = (descriptor_key, make(plan.lengths, torch.int64), make(plan.chunk_tensors, torch.int32),
                              make(plan.chunk_offsets,
                                   torch.int64), make(plan.chunk_starts,
                                                      torch.int64), make(plan.chunk_counts, torch.int64))

    _, lengths, chunk_tensors, chunk_offsets, chunk_starts, chunk_counts = stable_descriptors
    partial_sum_squares = torch.empty(len(plan.chunk_tensors), dtype=torch.float64, device=device)
    partial_invalid = torch.empty(len(plan.chunk_tensors), dtype=torch.int32, device=device)
    tensor_sum_squares = torch.empty(len(plan.gradients), dtype=torch.float64, device=device)
    tensor_invalid = torch.empty(len(plan.gradients), dtype=torch.int32, device=device)
    if plan.chunk_tensors:
        _chunk_gradient_statistics[(len(plan.chunk_tensors), )](pointers,
                                                                lengths,
                                                                chunk_tensors,
                                                                chunk_offsets,
                                                                partial_sum_squares,
                                                                partial_invalid,
                                                                BLOCK=_CHUNK_SIZE,
                                                                num_warps=8)
    _tensor_gradient_statistics[(len(plan.gradients), )](partial_sum_squares,
                                                         partial_invalid,
                                                         chunk_starts,
                                                         chunk_counts,
                                                         tensor_sum_squares,
                                                         tensor_invalid,
                                                         REDUCE_BLOCK=256,
                                                         num_warps=4)
    keepalive = (pointers, stable_descriptors, partial_sum_squares, partial_invalid)
    return tensor_sum_squares, tensor_invalid.bool(), keepalive


@dataclass
class _GradientStatisticsCache:
    plan: GradientStatisticsPlan
    sum_squares: torch.Tensor
    invalid: torch.Tensor
    selected: tuple
    keepalive: tuple
    consumed: set


class FusedGradientStatistics:

    def __init__(self, model_parallel_rank):
        self.model_parallel_rank = model_parallel_rank
        self.cache = None
        self.active_for_update = False
        self.last_fallback_reason = None
        self.refresh_count = 0

    @staticmethod
    def is_available():
        return _TRITON_AVAILABLE

    def refresh(self, gradient_groups, parameter_groups):
        previous = self.cache
        try:
            plan = build_gradient_statistics_plan(gradient_groups,
                                                  parameter_groups,
                                                  self.model_parallel_rank,
                                                  cached_plan=previous.plan if previous is not None else None)
        except ValueError as error:
            return self._fallback(str(error))

        fallback_reason = _preflight_plan(plan)
        if fallback_reason is not None:
            return self._fallback(fallback_reason)

        stable_descriptors = previous.keepalive[1] if previous is not None else None
        sum_squares, invalid, keepalive = _run_fused_gradient_statistics(plan, stable_descriptors)
        device = sum_squares.device
        if previous is not None and previous.plan.group_selected == plan.group_selected:
            selected = previous.selected
        else:
            selected = tuple(
                torch.tensor(indices, dtype=torch.int64, device=device) for indices in plan.group_selected)

        self.cache = _GradientStatisticsCache(replace(plan, gradients=()), sum_squares, invalid, selected, keepalive,
                                              set())
        self.active_for_update = True
        self.last_fallback_reason = None
        self.refresh_count += 1
        return invalid.any().reshape(1)

    def local_group_sum_squares(self, gradients, parameters):
        if not self.active_for_update or self.cache is None:
            raise RuntimeError("fused gradient statistics cache is not active for this update")
        try:
            group_index = self.cache.plan.group_keys.index((id(gradients), id(parameters)))
        except ValueError as error:
            raise RuntimeError("fused gradient statistics cache does not match the requested group") from error
        if group_index in self.cache.consumed:
            raise RuntimeError("fused gradient statistics for this group were already consumed")

        self.cache.consumed.add(group_index)
        indices = self.cache.selected[group_index]
        if not indices.numel():
            return torch.tensor(0.0, dtype=torch.float32, device=self.cache.sum_squares.device)

        per_tensor_norms = self.cache.sum_squares.index_select(0, indices).sqrt()
        return per_tensor_norms.square().sum().float()

    def _fallback(self, reason):
        self.cache = None
        self.active_for_update = False
        self.last_fallback_reason = reason
        return None
