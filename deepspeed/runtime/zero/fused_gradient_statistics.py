# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass, replace

import torch

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE
from deepspeed.ops.triton_ops.fused_gradient_statistics import run_fused_gradient_statistics
from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.runtime.utils import is_model_parallel_parameter

_IS_ROCM_PYTORCH = getattr(torch.version, "hip", None) is not None
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


def _availability_reason(device):
    device = torch.device(device)
    if device.type != "cuda":
        return "the device is not CUDA"
    if _IS_ROCM_PYTORCH:
        return "ROCm is unsupported"
    if not _TRITON_AVAILABLE:
        return "Triton is unavailable"

    major, minor = torch.cuda.get_device_capability(device)  #ignore-cuda
    if major < 8:
        return f"NVIDIA compute capability {major}.{minor} is below the required 8.0"
    return None


@dataclass
class _GradientStatisticsCache:
    plan: GradientStatisticsPlan
    sum_squares: torch.Tensor
    invalid: torch.Tensor
    selected: tuple
    keepalive: tuple
    consumed: set


class FusedGradientStatistics:

    def __init__(self, model_parallel_rank, device):
        self.model_parallel_rank = model_parallel_rank
        self.device = torch.device(device)
        self.cache = None
        self.active_for_update = False
        self.last_fallback_reason = None
        self.refresh_count = 0

    @staticmethod
    def unavailable_reason(device):
        return _availability_reason(device)

    @staticmethod
    def is_available(device):
        return _availability_reason(device) is None

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
        if fallback_reason is None:
            device = plan.gradients[0].device if plan.gradients else self.device
            fallback_reason = _availability_reason(device)
        if fallback_reason is not None:
            return self._fallback(fallback_reason)

        stable_descriptors = previous.keepalive[1] if previous is not None else None
        sum_squares, invalid, keepalive = run_fused_gradient_statistics(plan.gradients, plan.lengths,
                                                                        plan.chunk_tensors, plan.chunk_offsets,
                                                                        plan.chunk_starts, plan.chunk_counts,
                                                                        self.device, stable_descriptors)
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
