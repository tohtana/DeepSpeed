# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Optional, Set, Any, Tuple
from dataclasses import dataclass

import torch

import deepspeed.comm as dist
from deepspeed.utils import logger
from deepspeed.comm.torch import get_coalescing_manager
from deepspeed.utils.torch import required_torch_version
from deepspeed.accelerator import get_accelerator

from .config import UniversalOptimizerConfig

# https://github.com/NVIDIA/nccl/issues/413#issuecomment-720634194
COMM_PAD_BYTE_SIZE = 16


def log_rank0(message, log_level=logging.INFO):
    if not dist.is_initialized():
        raise RuntimeError("Distributed is not initialized")
    if dist.get_rank() == 0:
        logger.log(log_level, f"[r{dist.get_rank()}] {message}")


def log_all_ranks_sorted(message, log_level=logging.INFO):
    if not dist.is_initialized():
        raise RuntimeError("Distributed is not initialized")
    for rank in range(dist.get_world_size()):
        if rank == dist.get_rank():
            logger.log(log_level, f"[r{rank}] {message}")
        dist.barrier()


def tensor_to_short_string(t, num_elements=10):
    list_val = t.flatten().tolist()[:num_elements]
    list_val_str = [f"{x:.4f}" for x in list_val]
    return f"[{', '.join(list_val_str)}]"


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def ceil_to_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def aligned_per_shard_size(data_size: int, world_size: int, pad_size: int) -> int:
    per_shard_max = ceil_div(data_size, world_size)
    return ceil_to_multiple(per_shard_max, pad_size)


def ensure_same_dtype_in_param_group(param_group: Dict[str, Any]) -> None:
    if len(param_group['params']) == 0:
        return

    dtype = param_group['params'][0].dtype
    assert all(p.dtype == dtype for p in param_group['params']), \
        "All parameters in a param_group must have the same dtype."


def ensure_same_device_in_all_param_groups(param_groups: List[Dict[str, Any]]) -> torch.device:
    devices = set()
    for param_group in param_groups:
        if len(param_group['params']) == 0:
            continue
        devices.add(param_group['params'][0].device)
        assert all(p.device == param_group['params'][0].device for p in param_group['params']), \
            "All parameters in a param_group must be on the same device."
    assert len(devices) == 1, "All param_groups must be on the same device."
    return devices.pop()


def sharded_counts_and_totals(size: int, dtype: torch.dtype, world_size: int, pad_bytes: int = COMM_PAD_BYTE_SIZE):
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    elems_per_rank_nominal = (size + world_size - 1) // world_size

    # Calculate the padded sendcount for each rank
    bytes_per_rank = elems_per_rank_nominal * bytes_per_elem
    padded_bytes_per_rank = ceil_to_multiple(bytes_per_rank, pad_bytes)
    per_rank_sendcount = padded_bytes_per_rank // bytes_per_elem

    # Calculate the total padded elements across all ranks
    total_padded_bytes = padded_bytes_per_rank * world_size
    total_padded_elems = total_padded_bytes // bytes_per_elem

    return per_rank_sendcount, total_padded_elems


def param_group_dtype(param_group: Dict[str, Any]) -> torch.dtype:
    if len(param_group['params']) == 0:
        raise ValueError("param_group has no parameters.")
    return param_group['params'][0].dtype


@dataclass
class BufferRange:
    offset: int
    size: int
    padded_size: int


@dataclass
class ParamGroupDtypes:
    param_dtype: torch.dtype
    grad_accum_dtype: torch.dtype
    optimizer_dtype: torch.dtype


@dataclass
class ParamUpdateFlatBuffers:
    param_buffer: torch.Tensor  # unsharded
    # Mapping from a parameter to its buffer range in the global unsharded flattened buffer
    param_range_map_global: Dict[torch.Tensor, BufferRange]
    # Mapping from a parameter to its buffer range in the local sharded flattened buffer
    param_range_map_local: Dict[torch.Tensor, BufferRange]
    grad_acc_buffer: torch.Tensor  # sharded
    grad_comm_buffer: torch.Tensor  # sharded, dtype matches parameter dtype when not shared
    param_for_optimizer: torch.Tensor  # sharded
    per_rank_padded_numel: int
    grad_accum_dtype: torch.dtype
    param_dtype: torch.dtype
    shard_views: List['ParamUpdateShardBuffers']

    def _can_share(self, share_grad_and_comm_buffer: bool) -> bool:
        return share_grad_and_comm_buffer and self.grad_accum_dtype == self.param_dtype

    def allocate_grad_buffers(self, share_grad_and_comm_buffer: bool) -> None:
        can_share = self._can_share(share_grad_and_comm_buffer)
        device = self.param_buffer.device
        if self.per_rank_padded_numel == 0:
            empty_acc = torch.empty(0, dtype=self.grad_accum_dtype, device=device)
            self.grad_acc_buffer = empty_acc
            self.grad_comm_buffer = empty_acc if can_share else torch.empty(0, dtype=self.param_dtype, device=device)
        else:
            self.grad_acc_buffer = torch.zeros(self.per_rank_padded_numel, dtype=self.grad_accum_dtype, device=device)
            if can_share:
                self.grad_comm_buffer = self.grad_acc_buffer
            else:
                self.grad_comm_buffer = torch.zeros(self.per_rank_padded_numel, dtype=self.param_dtype, device=device)

        for shard in self.shard_views:
            shard.refresh_views(self, share_grad_and_comm_buffer)

    def release_grad_buffers(self, share_grad_and_comm_buffer: bool) -> None:
        can_share = self._can_share(share_grad_and_comm_buffer)
        device = self.param_buffer.device
        empty_acc = torch.empty(0, dtype=self.grad_accum_dtype, device=device)
        self.grad_acc_buffer = empty_acc
        self.grad_comm_buffer = self.grad_acc_buffer if can_share else torch.empty(
            0, dtype=self.param_dtype, device=device)
        for shard in self.shard_views:
            shard.refresh_views(self, share_grad_and_comm_buffer)


@dataclass
class ParamUpdateShardBuffers:
    size: int
    padded_size: int
    grad_acc_buffer: torch.Tensor  # sharded
    grad_comm_buffer: torch.Tensor  # sharded
    param_for_optimizer: torch.Tensor  # sharded
    param_buffer: torch.Tensor  # sharded view in param_buffer dtype
    grad_offset: int
    param_buffer_offset: int

    def refresh_views(self, flat_buffers: ParamUpdateFlatBuffers, share_grad_and_comm_buffer: bool) -> None:
        can_share = flat_buffers._can_share(share_grad_and_comm_buffer)

        if self.padded_size == 0 or flat_buffers.grad_acc_buffer.numel() == 0:
            self.grad_acc_buffer = flat_buffers.grad_acc_buffer.narrow(0, 0, 0)
        else:
            self.grad_acc_buffer = flat_buffers.grad_acc_buffer.narrow(0, self.grad_offset, self.padded_size)

        if self.padded_size == 0:
            self.grad_comm_buffer = flat_buffers.grad_comm_buffer.narrow(0, 0, 0)
        else:
            if can_share:
                self.grad_comm_buffer = self.grad_acc_buffer
            elif flat_buffers.grad_comm_buffer.numel() == 0:
                self.grad_comm_buffer = flat_buffers.grad_comm_buffer.narrow(0, 0, 0)
            else:
                self.grad_comm_buffer = flat_buffers.grad_comm_buffer.narrow(0, self.grad_offset, self.padded_size)

        # param_for_optimizer and param_buffer are static buffers, no need to refresh unless resized


class ReduceBucket:

    def __init__(self, dtype: torch.dtype, buffer_size: int, device: torch.device) -> None:
        self.buffer = None  # Lazy allocation
        self.size = buffer_size
        self.offset = 0
        self.dtype = dtype
        self.device = device

    def _ensure_allocated(self) -> None:
        """Lazily allocate buffer on first use"""
        if self.buffer is None:
            self.buffer = torch.empty(self.size, dtype=self.dtype, device=self.device)

    def get_size(self) -> int:
        return self.size

    def get_offset(self) -> int:
        return self.offset

    def get_buffer(self) -> torch.Tensor:
        self._ensure_allocated()
        return self.buffer

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def reserve(self, size: int) -> None:
        if size > self.size:
            self.buffer = torch.empty(size, dtype=self.dtype, device=self.device)
            self.size = size

    def allocate(self, numel: int) -> torch.Tensor:
        self._ensure_allocated()
        if self.offset + numel > self.size:
            raise RuntimeError("Buffer size exceeds the reduce bucket size")

        result = self.buffer[self.offset:self.offset + numel]
        self.offset += numel
        return result

    def should_flush(self, numel: int) -> bool:
        return self.offset > 0 and self.offset + numel > self.size

    def reset(self) -> None:
        self.offset = 0

    def release(self) -> None:
        """Release buffer memory"""
        self.buffer = None
        self.offset = 0


class CommDoubleBuffer:

    def __init__(self, dtype: torch.dtype, buffer_size: int, device: torch.device) -> None:
        self.buckets: List[ReduceBucket] = [
            ReduceBucket(dtype, buffer_size, device),
            ReduceBucket(dtype, buffer_size, device)
        ]
        self.events = [
            get_accelerator().Event(enable_timing=False, blocking=False),
            get_accelerator().Event(enable_timing=False, blocking=False)
        ]
        self.current_buffer_idx = 0
        self.size = buffer_size

    def get_size(self) -> int:
        return self.size

    def get_buffer(self) -> torch.Tensor:
        return self.buckets[self.current_buffer_idx]

    def get_event(self):
        return self.events[self.current_buffer_idx]

    def should_flush(self, size: int) -> bool:
        return self.buckets[self.current_buffer_idx].should_flush(size)

    def allocate(self, size: int) -> torch.Tensor:
        return self.buckets[self.current_buffer_idx].allocate(size)

    def swap(self, copy_stream) -> None:
        self.events[self.current_buffer_idx].record(copy_stream)
        self.buckets[self.current_buffer_idx].reset()
        self.current_buffer_idx = 1 - self.current_buffer_idx

    def release(self) -> None:
        """Release all buffer memory"""
        for bucket in self.buckets:
            bucket.release()
        self.current_buffer_idx = 0


@dataclass
class ReduceTask:
    param: torch.Tensor
    send_buf: torch.Tensor  # comm buffer, different from `grad`
    recv_buf: torch.Tensor
    grad_acc_buf: torch.Tensor
    data_size: int


@dataclass
class ReduceResult:
    param: torch.Tensor
    reduced_grad: torch.Tensor
    data_size: int


@dataclass
class GradientChunk:
    dtype: torch.dtype
    results: List[ReduceResult]
    param_group: Dict[str, Any]


class GradConversionDoubleBuffer:

    def __init__(self, dtype: torch.dtype, device: torch.device) -> None:
        self.dtype = dtype
        self.device = device
        self.buffers: List[torch.Tensor] = [
            torch.empty(0, dtype=dtype, device=device),
            torch.empty(0, dtype=dtype, device=device)
        ]
        self.current = 0

    def ensure_capacity(self, numel: int) -> None:
        if self.buffers[self.current].numel() < numel:
            self.buffers[self.current] = torch.empty(numel, dtype=self.dtype, device=self.device)

    def acquire(self) -> torch.Tensor:
        buffer = self.buffers[self.current]
        self.current = 1 - self.current
        return buffer

    def release(self) -> None:
        self.buffers = [
            torch.empty(0, dtype=self.dtype, device=self.device),
            torch.empty(0, dtype=self.dtype, device=self.device)
        ]
        self.current = 0


class ParamUpdateGroupContainer:
    """ A container of ParamUpdateGroup, each group is identified by (dtype, param_group).
    """

    def __init__(self, optimizer: torch.optim.Optimizer, device: torch.device, world_size: int, rank: int,
                 grad_accum_dtype: Optional[torch.dtype], optimizer_dtype: Optional[torch.dtype],
                 share_grad_and_comm_buffer: bool) -> None:

        self.optimizer = optimizer
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.grad_accum_dtype = grad_accum_dtype
        self.optimizer_dtype = optimizer_dtype
        self.share_grad_and_comm_buffer = share_grad_and_comm_buffer

        # Initialize buffers for param update for each param group
        self.param_update_buffers: List[ParamUpdateFlatBuffers] = []
        self.sharded_param_groups: List[Dict[str, Any]] = []
        for param_group in optimizer.param_groups:
            if len(param_group['params']) == 0:
                continue

            # Create contiguous buffers for the param group
            sharded_contiguous_buffer = self._init_param_update_buffers(param_group)
            self.param_update_buffers.append(sharded_contiguous_buffer)

            # Create a param group containing the contiguous buffers.
            # Used for the optimizer.
            sharded_param_group = copy.copy(param_group)
            contiguous_param = sharded_contiguous_buffer.param_for_optimizer
            # contiguous_param.grad = sharded_contiguous_buffer.grad_for_optimizer
            sharded_param_group['params'] = [contiguous_param]
            self.sharded_param_groups.append(sharded_param_group)

        # Create a map from param to its buffer in the contiguous buffer
        self.param_buffer_map: Dict[torch.Tensor, ParamUpdateShardBuffers] = {}
        self.param_to_group_buffer: Dict[torch.Tensor, ParamUpdateFlatBuffers] = {}
        for pg_buffers in self.param_update_buffers:
            for param, local_map in pg_buffers.param_range_map_local.items():
                offset, size, padded_size = local_map.offset, local_map.size, local_map.padded_size
                map_global = pg_buffers.param_range_map_global[param]
                local_shard_offset = map_global.offset + padded_size * self.rank
                if pg_buffers.grad_acc_buffer.numel() == 0 or padded_size == 0:
                    grad_acc_view = pg_buffers.grad_acc_buffer.narrow(0, 0, 0)
                else:
                    grad_acc_view = pg_buffers.grad_acc_buffer.narrow(0, offset, padded_size)

                if pg_buffers.grad_comm_buffer is pg_buffers.grad_acc_buffer:
                    grad_comm_view = grad_acc_view
                elif pg_buffers.grad_comm_buffer.numel() == 0 or padded_size == 0:
                    grad_comm_view = pg_buffers.grad_comm_buffer.narrow(0, 0, 0)
                else:
                    grad_comm_view = pg_buffers.grad_comm_buffer.narrow(0, offset, padded_size)

                param_for_optimizer_view = pg_buffers.param_for_optimizer.narrow(0, offset, padded_size)
                param_buffer_view = pg_buffers.param_buffer.narrow(0, local_shard_offset, padded_size)

                shard_buffers = ParamUpdateShardBuffers(size=size,
                                                        padded_size=padded_size,
                                                        grad_acc_buffer=grad_acc_view,
                                                        grad_comm_buffer=grad_comm_view,
                                                        param_for_optimizer=param_for_optimizer_view,
                                                        param_buffer=param_buffer_view,
                                                        grad_offset=offset,
                                                        param_buffer_offset=local_shard_offset)

                pg_buffers.shard_views.append(shard_buffers)
                self.param_buffer_map[param] = shard_buffers
                self.param_to_group_buffer[param] = pg_buffers

        self.grad_buffers_allocated: bool = False

    def allocate_grad_buffers(self) -> None:
        if self.grad_buffers_allocated:
            return
        for buffers in self.param_update_buffers:
            buffers.allocate_grad_buffers(self.share_grad_and_comm_buffer)
        self.grad_buffers_allocated = True

    def release_grad_buffers(self) -> None:
        if not self.grad_buffers_allocated:
            return
        for buffers in self.param_update_buffers:
            buffers.release_grad_buffers(self.share_grad_and_comm_buffer)
        self.grad_buffers_allocated = False

    def _init_param_update_buffers(self, param_group: Dict[str, Any]) -> ParamUpdateFlatBuffers:
        param_group_dtypes = self._param_group_dtypes(param_group)

        # First we create a flattened buffer for all parameters in the param group
        param_range_map_global: Dict[torch.Tensor, BufferRange] = {}
        padded_total_numel = 0
        # Create a range map for the flattened buffer.
        # Try to align shards to both the comm pad size and the world size.
        for p in param_group['params']:
            per_rank_padded_numel = aligned_per_shard_size(p.numel(), self.world_size, COMM_PAD_BYTE_SIZE)
            total_padded_numel = per_rank_padded_numel * self.world_size
            param_range_map_global[p] = BufferRange(offset=padded_total_numel,
                                                    size=p.numel(),
                                                    padded_size=total_padded_numel)
            padded_total_numel += total_padded_numel

        # Stage the parameter data on CPU so we can release the original GPU storage before
        # allocating the flattened buffer. This mirrors the ZeRO stage-1 flow and avoids the
        # temporary 2x model footprint during initialization.
        cpu_param_copies: Dict[torch.Tensor, torch.Tensor] = {}
        for p in param_group['params']:
            cpu_param_copies[p] = p.data.detach().cpu()
            # Replace with an empty tensor on device to release the original storage.
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)

        flat_param_buffer = torch.empty(padded_total_numel, dtype=param_group_dtypes.param_dtype, device=self.device)

        # Remap parameters to the flat buffer and restore their values from the CPU staging area.
        for p in param_group['params']:
            offset = param_range_map_global[p].offset
            size = param_range_map_global[p].size
            cpu_copy = cpu_param_copies[p]
            param_view = flat_param_buffer[offset:offset + size].view_as(cpu_copy)
            param_view.copy_(cpu_copy)
            p.data = param_view

        cpu_param_copies.clear()

        # Only Z1/Z2: Broadcast the param buffer to all ranks, just in case
        # broadcast param to all rank
        dist.broadcast(flat_param_buffer, src=0)

        # Create the grad accumulation buffer
        # assert padded_total_numel % self.world_size == 0, f"padded_total_numel {padded_total_numel} must be a multiple of world_size {self.world_size}"
        # per_rank_padded_numel = padded_total_numel // self.world_size

        # Create a range map for per-rank data:
        # Each rank has a flattened buffer that concatenates shards of *all* parameters.
        # Note that this is different from the traditional layout of DeepSpeed ZeRO, which partitions the flattened buffer.
        # In traditional ZeRO layout:
        # ---------------------------------------------------
        # |       p#0     | p#1 |   p#2   |       p#3       |
        # |        r0               |          r1           |
        # ---------------------------------------------------
        # In this layout:
        # ---------------------------------------------------
        # |       p#0     | p#1 |   p#2   |       p#3       |
        # |   r0   |  r1  |r0|r1| r0 | r1 |   r0   |   r1   |
        # ---------------------------------------------------

        # We need this layout to support reduce-scatter. To avoid many reduce-scatter calls, we need _coalescing_manager (i.e. group call in NCCL).

        # We also need the mapping from param to range in the per-rank buffer, like this:
        # |    p#0[r0]   |p#1[r0]| p#2[r0] |  p#3[r0]  |
        # |    p#0[r1]   |p#1[r1]| p#2[r1] |  p#3[r1]  |

        param_range_map_local: Dict[torch.Tensor, BufferRange] = {}
        offset = 0
        per_rank_padded_numel = 0
        for p in param_group['params']:
            assert param_range_map_global[
                p].padded_size % self.world_size == 0, f"padded_size {param_range_map_global[p].padded_size} must be a multiple of world_size {self.world_size}"
            padded_shard_size = param_range_map_global[p].padded_size // self.world_size
            # Record the size except the padding. `shard_size` might be zero if the rank only contains a padding region.
            shard_size = min(max(param_range_map_global[p].size - padded_shard_size * self.rank, 0), padded_shard_size)
            param_range_map_local[p] = BufferRange(offset=offset, size=shard_size, padded_size=padded_shard_size)
            offset += padded_shard_size

        # *Per-rank* flattened grad accumulation buffer
        per_rank_padded_numel = offset
        grad_acc_buffer = torch.empty(0, dtype=param_group_dtypes.grad_accum_dtype, device=self.device)

        if param_group_dtypes.grad_accum_dtype == param_group_dtypes.param_dtype:
            grad_comm_buffer = grad_acc_buffer
        else:
            grad_comm_buffer = torch.empty(0, dtype=param_group_dtypes.param_dtype, device=self.device)

        param_for_optimizer = torch.empty(per_rank_padded_numel,
                                          dtype=param_group_dtypes.optimizer_dtype,
                                          device=self.device)
        for p in param_group['params']:
            offset_global = param_range_map_global[p].offset
            padded_size_global = param_range_map_global[p].padded_size
            padded_shard_size = padded_size_global // self.world_size
            shard_offset = offset_global + padded_shard_size * self.rank
            copy_src = flat_param_buffer[shard_offset:shard_offset + padded_shard_size]

            offset_local = param_range_map_local[p].offset
            padded_size_local = param_range_map_local[p].padded_size
            param_for_optimizer[offset_local:offset_local + padded_size_local].copy_(copy_src.view(-1))

        return ParamUpdateFlatBuffers(param_buffer=flat_param_buffer,
                                      param_range_map_global=param_range_map_global,
                                      param_range_map_local=param_range_map_local,
                                      grad_acc_buffer=grad_acc_buffer,
                                      grad_comm_buffer=grad_comm_buffer,
                                      param_for_optimizer=param_for_optimizer,
                                      per_rank_padded_numel=per_rank_padded_numel,
                                      grad_accum_dtype=param_group_dtypes.grad_accum_dtype,
                                      param_dtype=param_group_dtypes.param_dtype,
                                      shard_views=[])

    def _param_group_dtypes(self, param_group: Dict[str, Any]) -> ParamGroupDtypes:
        param_dtype = param_group_dtype(param_group)
        grad_accum_dtype = param_dtype if self.grad_accum_dtype is None else self.grad_accum_dtype
        optimizer_dtype = param_dtype if self.optimizer_dtype is None else self.optimizer_dtype
        return ParamGroupDtypes(param_dtype=param_dtype,
                                grad_accum_dtype=grad_accum_dtype,
                                optimizer_dtype=optimizer_dtype)


class UniversalOptimizer(ABC):

    def __init__(self, optimizer: torch.optim.Optimizer, config: UniversalOptimizerConfig, reduce_bucket_size: int,
                 clip_grad: float) -> None:

        # for `register_post_accumulate_grad_hook` and `_coalescing_manager`
        assert required_torch_version(min_version=2.1), "UniversalOptimizer requires PyTorch 2.1 or higher."

        self.base_optimizer: torch.optim.Optimizer = optimizer

        self.force_model_dtype: torch.dtype = config.force_model_dtype
        self.zero3_allgather_dtype: torch.dtype = config.zero3_allgather_dtype
        self.reduce_dtype: torch.dtype = config.reduce_dtype
        self.grad_accum_dtype: torch.dtype = config.grad_accum_dtype
        self.optimizer_dtype: torch.dtype = config.optimizer_dtype
        self.clip_grad: float = clip_grad
        self._global_grad_norm: float = 0.0

        self.is_gradient_accumulation_boundary: bool = True

        self.reduce_op = dist.ReduceOp.AVG
        self.gradient_chunk_size = config.gradient_chunk_size

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # Force model dtype if necessary
        if self.force_model_dtype is not None:
            for group in self.base_optimizer.param_groups:
                for p in group['params']:
                    p.data = p.data.to(self.force_model_dtype)

        # Validations
        for param_group in self.base_optimizer.param_groups:
            ensure_same_dtype_in_param_group(param_group)
        self.device = ensure_same_device_in_all_param_groups(self.base_optimizer.param_groups)

        # Use communication buffer per dtype
        self.comm_buffers: Dict[torch.dtype, CommDoubleBuffer] = self._create_comm_buffers(reduce_bucket_size)
        self.grad_conversion_buffers: Dict[torch.dtype, GradConversionDoubleBuffer] = {}

        self.param_update_group_container = ParamUpdateGroupContainer(
            optimizer=self.base_optimizer,
            device=self.device,
            world_size=self.world_size,
            rank=self.rank,
            grad_accum_dtype=self.grad_accum_dtype,
            optimizer_dtype=self.optimizer_dtype,
            share_grad_and_comm_buffer=self._share_grad_and_comm_buffer())
        self.param_buffer_map = self.param_update_group_container.param_buffer_map
        self.sharded_param_groups = self.param_update_group_container.sharded_param_groups
        self.param_update_buffers = self.param_update_group_container.param_update_buffers

        self.param_to_param_group_index: Dict[torch.Tensor, int] = {}
        for group_idx, param_group in enumerate(self.base_optimizer.param_groups):
            for p in param_group['params']:
                self.param_to_param_group_index[p] = group_idx

        self.gradient_hook_handles = self._create_gradient_handling_hooks()

        self.reduce_tasks: Dict[torch.dtype, List[ReduceTask]] = defaultdict(list)
        self.reduce_results: Dict[torch.dtype, List[ReduceResult]] = defaultdict(list)

        self.comp_stream = get_accelerator().current_stream()
        self.rs_stream = get_accelerator().Stream(priority=-1)
        self.copy_stream = get_accelerator().Stream(priority=-1)

        self.rs_comp_done_events = defaultdict(get_accelerator().Event)
        self.rs_copy_done_events = defaultdict(get_accelerator().Event)

    @abstractmethod
    def _share_grad_and_comm_buffer(self) -> bool:
        """Indicate whether gradient accumulation and communication buffers can be shared."""

    @abstractmethod
    def _should_process_gradient(self, param: torch.Tensor) -> bool:
        """Return True if we should queue gradient communication for `param`."""

    @abstractmethod
    def _accumulate_into_grad_acc_buf(self, task: 'ReduceTask') -> None:
        """Combine the reduce-scatter result into the accumulation buffer."""

    def _reset_grad_accum_buffers(self) -> None:
        """Hook for subclasses to reset accumulation buffers during zero_grad."""
        pass

    @abstractmethod
    def _should_clear_param_grad(self) -> bool:
        """Return True if `param.grad` should be cleared after the hook."""

    @abstractmethod
    def _reduce_and_sqrt_grad_norm(self, norm_accum: torch.Tensor) -> torch.Tensor:
        """Return the L2 norm for gradient clipping, including any collective ops."""

    def _chunk_reduce_results(self) -> List[GradientChunk]:
        gradient_chunks: List[GradientChunk] = []

        for dtype, results in self.reduce_results.items():
            if not results:
                continue

            grouped_results: Dict[int, List[ReduceResult]] = defaultdict(list)
            group_order: List[int] = []

            for result in results:
                param_group_index = self.param_to_param_group_index.get(result.param)
                if param_group_index is None:
                    raise RuntimeError("Parameter missing from param group mapping during chunking")
                if param_group_index not in grouped_results:
                    group_order.append(param_group_index)
                grouped_results[param_group_index].append(result)

            for param_group_index in group_order:
                group_results = grouped_results[param_group_index]
                if not group_results:
                    continue

                chunk_results: List[ReduceResult] = []
                chunk_params: List[torch.Tensor] = []
                chunk_numel = 0

                for result in group_results:
                    param_buffers = self.param_buffer_map[result.param]
                    param_tensor = param_buffers.param_for_optimizer
                    shard_numel = result.data_size

                    if shard_numel == 0:
                        continue

                    if chunk_results and chunk_numel + shard_numel > self.gradient_chunk_size:
                        gradient_chunks.append(
                            self._build_gradient_chunk(dtype, param_group_index, chunk_results, chunk_params))
                        chunk_results = []
                        chunk_params = []
                        chunk_numel = 0

                    chunk_results.append(result)
                    chunk_params.append(param_tensor)
                    chunk_numel += shard_numel

                if chunk_results:
                    gradient_chunks.append(
                        self._build_gradient_chunk(dtype, param_group_index, chunk_results, chunk_params))

        self.reduce_results.clear()
        return gradient_chunks

    def _build_gradient_chunk(self, dtype: torch.dtype, param_group_index: int, results: List[ReduceResult],
                              params: List[torch.Tensor]) -> GradientChunk:
        base_group = self.base_optimizer.param_groups[param_group_index]
        chunk_param_group = dict(base_group)
        chunk_param_group['params'] = params
        chunk_dtype = dtype if params else results[0].reduced_grad.dtype
        return GradientChunk(dtype=chunk_dtype, results=results, param_group=chunk_param_group)

    def _get_conversion_buffer(self, dtype: torch.dtype) -> GradConversionDoubleBuffer:
        if dtype not in self.grad_conversion_buffers:
            self.grad_conversion_buffers[dtype] = GradConversionDoubleBuffer(dtype=dtype, device=self.device)
        return self.grad_conversion_buffers[dtype]

    def gradient_hook(self, param):
        if not self._should_process_gradient(param):
            return

        assert param.dtype in self.comm_buffers, f"Param dtype {param.dtype} not in comm buffers {list(self.comm_buffers.keys())}"
        comm_buffer = self.comm_buffers[param.dtype]

        if comm_buffer.should_flush(param.numel()):
            self.flush_reduce_bucket(param.dtype)

        if param.numel() > comm_buffer.get_size():
            # extend buckets
            get_accelerator().current_stream().synchronize()
            comm_buffer.buckets[comm_buffer.current_buffer_idx].reserve(param.numel())

        reduce_in_buffer = comm_buffer.allocate(ceil_to_multiple(param.numel(), COMM_PAD_BYTE_SIZE))

        # This ensures the order of reduce_scatter -> copy
        # Without this block, copy may start while reduce_scatter is still running
        comm_buffer.get_event().wait(self.comp_stream)

        param_buffers = self.param_buffer_map[param]
        recv_buf = param_buffers.grad_comm_buffer
        grad_acc_buf = param_buffers.grad_acc_buffer
        shard_size = param_buffers.size
        self.reduce_tasks[param.dtype].append(ReduceTask(param, reduce_in_buffer, recv_buf, grad_acc_buf, shard_size))

        self.rs_comp_done_events[param].record(self.comp_stream)
        self.rs_comp_done_events[param].wait(self.copy_stream)
        with get_accelerator().stream(self.copy_stream):
            reduce_in_buffer.view(-1).narrow(0, 0, param.numel()).copy_(param.grad.view(-1), non_blocking=True)
            param.grad.record_stream(self.copy_stream)
            self.rs_copy_done_events[param].record(self.copy_stream)

        if self._should_clear_param_grad():
            param.grad.data = torch.empty(0, dtype=param.dtype,
                                          device=self.device)  # free the original grad to reduce memory usage

    def flush_reduce_bucket(self, dtype: torch.dtype):
        if dtype not in self.reduce_tasks:
            return

        self._block_copy_events(dtype)

        with get_coalescing_manager(
                group=None,
                device=self.device,
                async_op=True,
        ) as cm:
            for t in self.reduce_tasks[dtype]:
                dist.reduce_scatter_tensor(
                    t.recv_buf,
                    t.send_buf,
                    op=self.reduce_op,
                    group=None,
                    async_op=True,
                )

        with get_accelerator().stream(self.copy_stream):
            cm.wait()

            for t in self.reduce_tasks[dtype]:
                self._accumulate_into_grad_acc_buf(t)
                padded_numel = t.grad_acc_buf.numel()
                if t.data_size < padded_numel:
                    t.grad_acc_buf.narrow(0, t.data_size, padded_numel - t.data_size).zero_()

            for t in self.reduce_tasks[dtype]:
                self.reduce_results[dtype].append(ReduceResult(t.param, t.grad_acc_buf, t.data_size))

        for t in self.reduce_tasks[dtype]:
            self.rs_copy_done_events[t.param].record(self.copy_stream)

        self.reduce_tasks[dtype].clear()
        self.comm_buffers[dtype].swap(self.copy_stream)

    def _backward_epilogue(self):
        for dtype in self.comm_buffers.keys():
            self.flush_reduce_bucket(dtype)

    def backward(self, loss, retain_graph=False) -> None:
        self.param_update_group_container.allocate_grad_buffers()
        loss.backward(retain_graph=retain_graph)
        self._backward_epilogue()

    def step(self, *args, **kwargs):
        self.copy_stream.synchronize()

        if not self.reduce_results:
            return

        gradient_chunks = self._chunk_reduce_results()
        original_param_groups = self.base_optimizer.param_groups
        issued_cast_copies = False

        grad_clip_coef: Optional[float] = None
        norm_accum = torch.zeros(1, dtype=torch.float32, device=self.device)

        for buffers in self.param_update_buffers:
            grad_buffer = buffers.grad_acc_buffer
            if grad_buffer.numel() == 0:
                continue
            grad_flat = grad_buffer.view(-1)
            grad_view = grad_flat if grad_flat.dtype == torch.float32 else grad_flat.float()
            norm_accum += torch.dot(grad_view, grad_view)

        global_grad_norm = self._reduce_and_sqrt_grad_norm(norm_accum)
        self._global_grad_norm = global_grad_norm.item()

        if self.clip_grad > 0.0:
            clip_coef = self.clip_grad / (self._global_grad_norm + 1e-6)
            if clip_coef < 1.0:
                grad_clip_coef = float(clip_coef)

        # Calculate total conversion needs across all chunks
        total_conversion_needs: Dict[torch.dtype, int] = defaultdict(int)
        for chunk in gradient_chunks:
            chunk_params = chunk.param_group['params']
            for result, param_tensor in zip(chunk.results, chunk_params):
                if result.reduced_grad.dtype != param_tensor.dtype:
                    total_conversion_needs[param_tensor.dtype] += param_tensor.numel()

        # Allocate all conversion buffers at once
        global_buffer_state: Dict[torch.dtype, Tuple[torch.Tensor, int]] = {}
        for dtype, required_numel in total_conversion_needs.items():
            buffer_mgr = self._get_conversion_buffer(dtype)
            buffer_mgr.ensure_capacity(required_numel)
            global_buffer_state[dtype] = (buffer_mgr.acquire(), 0)

        # OPTIMIZATION: Overlap gradient copy with optimizer step by processing chunks in pipeline
        # For each chunk:
        #   1. Copy gradients for chunk N on copy_stream (overlaps with previous chunk's compute)
        #   2. Wait for copy to complete
        #   3. Run optimizer.step() for chunk N on default stream
        # This hides gradient copy latency behind compute

        for chunk in gradient_chunks:
            chunk_params = chunk.param_group['params']

            # Start copying gradients for this chunk on copy_stream
            # This overlaps with the previous chunk's optimizer.step() if any
            with get_accelerator().stream(self.copy_stream):
                for result, param_tensor in zip(chunk.results, chunk_params):
                    grad_tensor = result.reduced_grad
                    grad_flat = grad_tensor.view(-1)

                    if grad_clip_coef is not None:
                        grad_flat.mul_(grad_clip_coef)

                    if grad_tensor.dtype != param_tensor.dtype:
                        buffer_tensor, offset = global_buffer_state[param_tensor.dtype]
                        numel = param_tensor.numel()
                        slice_view = buffer_tensor.narrow(0, offset, numel)
                        slice_view.copy_(grad_flat, non_blocking=True)
                        global_buffer_state[param_tensor.dtype] = (buffer_tensor, offset + numel)
                        grad_src = slice_view
                    else:
                        grad_src = grad_flat

                    param_tensor.grad = grad_src.view_as(param_tensor)

            # Wait for this chunk's gradient copy to complete before running optimizer
            copy_event = get_accelerator().Event(enable_timing=False, blocking=False)
            copy_event.record(self.copy_stream)
            copy_event.wait(get_accelerator().current_stream())

            # Run optimizer step for this chunk on default stream
            # While this runs, the next chunk's gradients can be copied in parallel!
            chunk_param_group = chunk.param_group
            self.base_optimizer.param_groups = [chunk_param_group]
            self.base_optimizer.step(*args, **kwargs)

            for param_tensor in chunk_params:
                param_tensor.grad = None

        # Copy back updated parameters
        step_event = get_accelerator().Event(enable_timing=False, blocking=False)
        step_event.record(get_accelerator().current_stream())

        with get_accelerator().stream(self.copy_stream):
            step_event.wait(self.copy_stream)
            for chunk in gradient_chunks:
                for result in chunk.results:
                    param_buffers = self.param_buffer_map[result.param]
                    if param_buffers.param_buffer.numel() == 0:
                        continue
                    if param_buffers.param_for_optimizer.dtype == param_buffers.param_buffer.dtype:
                        continue
                    param_buffers.param_buffer.copy_(param_buffers.param_for_optimizer, non_blocking=True)
                    issued_cast_copies = True

        self.base_optimizer.param_groups = original_param_groups
        for buffer_mgr in self.grad_conversion_buffers.values():
            buffer_mgr.release()

        if issued_cast_copies:
            pre_gather_event = get_accelerator().Event(enable_timing=False, blocking=False)
            pre_gather_event.record(self.copy_stream)
            pre_gather_event.wait(self.comp_stream)

        with get_coalescing_manager(
                group=None,
                device=self.device,
                async_op=True,
        ) as cm:
            for buffers in self.param_update_buffers:
                for p in buffers.param_range_map_global.keys():
                    map_global = buffers.param_range_map_global[p]
                    map_local = buffers.param_range_map_local[p]

                    padded_shard_size = map_local.padded_size
                    if padded_shard_size == 0:
                        continue

                    local_shard_offset = map_global.offset + padded_shard_size * self.rank

                    if buffers.param_for_optimizer.dtype == buffers.param_buffer.dtype:
                        gather_src = buffers.param_for_optimizer[map_local.offset:map_local.offset + padded_shard_size]
                    else:
                        gather_src = buffers.param_buffer[local_shard_offset:local_shard_offset + padded_shard_size]

                    dist.all_gather_into_tensor(
                        buffers.param_buffer[map_global.offset:map_global.offset + map_global.padded_size], gather_src)
        cm.wait()

        self.param_update_group_container.release_grad_buffers()

        # Release communication buffers after step to save memory
        for comm_buffer in self.comm_buffers.values():
            comm_buffer.release()

    def zero_grad(self, *args, **kwargs):
        self.base_optimizer.zero_grad(*args, **kwargs)
        self._reset_grad_accum_buffers()
        self.param_update_group_container.release_grad_buffers()

    def clear_gradient_hooks(self):
        for handle in self.gradient_hook_handles:
            handle.remove()

    def _create_gradient_handling_hooks(self):
        hook_handles = []

        for param_group in self.base_optimizer.param_groups:
            for param in param_group['params']:
                if param.requires_grad:
                    hook_handles.append(param.register_post_accumulate_grad_hook(self.gradient_hook))

        return hook_handles

    def _create_comm_buffers(self, reduce_bucket_size: int) -> Dict[torch.dtype, CommDoubleBuffer]:
        dtypes: Set[torch.dtype] = set()
        for param_group in self.base_optimizer.param_groups:
            if len(param_group['params']) == 0:
                continue
            dtypes.add(param_group_dtype(param_group))

        # Use communication buffer per dtype
        comm_buffers: Dict[torch.dtype, CommDoubleBuffer] = {}
        for dtype in dtypes:
            comm_buffers[dtype] = CommDoubleBuffer(dtype=dtype, buffer_size=reduce_bucket_size, device=self.device)
        return comm_buffers

    def _block_copy_events(self, dtype: torch.dtype):
        if dtype not in self.reduce_tasks:
            return

        for t in self.reduce_tasks[dtype]:
            copy_done_event = self.rs_copy_done_events[t.param]
            copy_done_event.wait(self.comp_stream)

    #############################################################################################
    # DeepSpeed engine accesses these properties
    @property
    def state(self):
        return self.base_optimizer.state

    @state.setter
    def state(self, value):
        self.base_optimizer.state = value

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.base_optimizer.param_groups = value

    @property
    def loss_scale(self):
        return self.base_optimizer.loss_scale

    @loss_scale.setter
    def loss_scale(self, value):
        self.base_optimizer.loss_scale = value

    @property
    def cur_scale(self):
        return self.base_optimizer.cur_scale

    @cur_scale.setter
    def cur_scale(self, value):
        self.base_optimizer.cur_scale = value


class UniversalOptimizerZ1(UniversalOptimizer):

    def __call__(self, optimizer: torch.optim.Optimizer, config: UniversalOptimizerConfig, reduce_bucket_size: int,
                 clip_grad: float):
        super().__call__(optimizer, config, reduce_bucket_size, clip_grad)

    def _share_grad_and_comm_buffer(self) -> bool:
        return True

    def _should_process_gradient(self, param: torch.Tensor) -> bool:
        return self.is_gradient_accumulation_boundary

    def _accumulate_into_grad_acc_buf(self, task: ReduceTask) -> None:
        if task.grad_acc_buf is task.recv_buf or task.grad_acc_buf.numel() == 0:
            return
        task.grad_acc_buf.copy_(task.recv_buf, non_blocking=True)

    def _should_clear_param_grad(self) -> bool:
        return self.is_gradient_accumulation_boundary

    def _reduce_and_sqrt_grad_norm(self, norm_accum: torch.Tensor) -> torch.Tensor:
        return norm_accum.sqrt()


class UniversalOptimizerZ2(UniversalOptimizer):

    def _share_grad_and_comm_buffer(self) -> bool:
        return False

    def _should_process_gradient(self, param: torch.Tensor) -> bool:
        return True

    def _accumulate_into_grad_acc_buf(self, task: ReduceTask) -> None:
        if task.grad_acc_buf is task.recv_buf or task.grad_acc_buf.numel() == 0:
            return

        if task.grad_acc_buf.dtype == task.recv_buf.dtype:
            task.grad_acc_buf.add_(task.recv_buf)
        else:
            task.grad_acc_buf.add_(task.recv_buf.to(dtype=task.grad_acc_buf.dtype))

    def _reset_grad_accum_buffers(self) -> None:
        for buffers in self.param_update_buffers:
            buffers.grad_acc_buffer.zero_()

    def _should_clear_param_grad(self) -> bool:
        return True

    def _reduce_and_sqrt_grad_norm(self, norm_accum: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(norm_accum, op=dist.ReduceOp.SUM)
        return norm_accum.sqrt()


def configure_universal_optimizer(optimizer: torch.optim.Optimizer, config: UniversalOptimizerConfig,
                                  reduce_bucket_size: int, zero_stage: int, clip_grad: float):
    if zero_stage == 1:
        optimizer_cls = UniversalOptimizerZ1
    elif zero_stage == 2:
        optimizer_cls = UniversalOptimizerZ2
    else:
        raise ValueError(f"Universal optimizer does not yet support ZeRO stage {zero_stage}.")

    return optimizer_cls(optimizer, config, reduce_bucket_size, clip_grad)
