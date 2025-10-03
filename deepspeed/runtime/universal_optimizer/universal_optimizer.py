# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass

import torch

import deepspeed.comm as dist
from deepspeed.utils import logger
from deepspeed.comm.torch import get_coalescing_manager
from deepspeed.utils.torch import required_torch_version
from deepspeed.accelerator import get_accelerator

from .config import UniversalOptimizerConfig

# https://github.com/NVIDIA/nccl/issues/413#issuecomment-720634194
COMM_PAD_BYTE_SIZE = 32


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
    # Mapping from a paramteer to its buffer range in the local sharded flattened buffer
    param_range_map_local: Dict[torch.Tensor, BufferRange]
    grad_acc_buffer: torch.Tensor  # sharded
    grad_comm_buffer: torch.Tensor  # sharded, dtype matches parameter dtype
    grad_for_optimizer: torch.Tensor  # sharded
    param_for_optimizer: torch.Tensor  # sharded


@dataclass
class ParamUpdateShardBuffers:
    size: int
    padded_size: int
    grad_acc_buffer: torch.Tensor  # sharded
    grad_comm_buffer: torch.Tensor  # sharded
    grad_for_optimizer: torch.Tensor  # sharded
    param_for_optimizer: torch.Tensor  # sharded


class ReduceBucket:

    def __init__(self, dtype: torch.dtype, buffer_size: int, device: torch.device) -> None:
        self.buffer = torch.empty(buffer_size, dtype=dtype, device=device)
        self.size = buffer_size
        self.offset = 0
        self.dtype = dtype

    def get_size(self) -> int:
        return self.size

    def get_offset(self) -> int:
        return self.offset

    def get_buffer(self) -> torch.Tensor:
        return self.buffer

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def reserve(self, size: int) -> None:
        if size > self.size:
            self.buffer = torch.empty(size, dtype=self.dtype, device=self.buffer.device)
            self.size = size

    def allocate(self, numel: int) -> torch.Tensor:
        if self.offset + numel > self.size:
            raise RuntimeError("Buffer size exceeds the reduce bucket size")

        result = self.buffer[self.offset:self.offset + numel]
        self.offset += numel
        return result

    def should_flush(self, numel: int) -> bool:
        return self.offset > 0 and self.offset + numel > self.size

    def reset(self) -> None:
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

    def flush(self) -> None:
        self.swap()
        raise NotImplementedError("flush is not implemented yet.")

    def swap(self, copy_stream) -> None:
        self.events[self.current_buffer_idx].record(copy_stream)
        self.buckets[self.current_buffer_idx].reset()
        self.current_buffer_idx = 1 - self.current_buffer_idx


@dataclass
class ReduceTask:
    param: torch.Tensor
    grad: torch.Tensor
    send_buf: torch.Tensor  # comm buffer, different from `grad`
    recv_buf: torch.Tensor
    grad_acc_buf: torch.Tensor
    opt_grad: torch.Tensor  # optimizer grad, may be different from `recv_buf`


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
            contiguous_param.grad = sharded_contiguous_buffer.grad_for_optimizer
            sharded_param_group['params'] = [contiguous_param]
            self.sharded_param_groups.append(sharded_param_group)

        # Create a map from param to its buffer in the contiguous buffer
        self.param_buffer_map: Dict[torch.Tensor, ParamUpdateShardBuffers] = {}
        for pg_buffers in self.param_update_buffers:
            for param, local_map in pg_buffers.param_range_map_local.items():
                offset, size, padded_size = local_map.offset, local_map.size, local_map.padded_size
                grad_acc_buffer = pg_buffers.grad_acc_buffer[offset:offset + padded_size]
                grad_comm_buffer = pg_buffers.grad_comm_buffer[offset:offset + padded_size]
                grad_for_optimizer = pg_buffers.grad_for_optimizer[offset:offset + padded_size]
                param_for_optimizer = pg_buffers.param_for_optimizer[offset:offset + padded_size]
                self.param_buffer_map[param] = ParamUpdateShardBuffers(size=size,
                                                                       padded_size=padded_size,
                                                                       grad_acc_buffer=grad_acc_buffer,
                                                                       grad_comm_buffer=grad_comm_buffer,
                                                                       grad_for_optimizer=grad_for_optimizer,
                                                                       param_for_optimizer=param_for_optimizer)

    def _init_param_update_buffers(self, param_group: Dict[str, Any]) -> ParamUpdateFlatBuffers:
        param_group_dtypes = self._param_group_dtypes(param_group)

        # First we create a flattened buffer for all parameters in the param group
        param_range_map_global: Dict[torch.Tensor, BufferRange] = {}
        padded_total_numel = 0
        # Create a range map for the flattened buffer.
        # Try to align shards to both the comm pad size and the world size.
        for p in param_group['params']:
            padded_numel = aligned_per_shard_size(p.numel(), self.world_size, COMM_PAD_BYTE_SIZE)
            param_range_map_global[p] = BufferRange(offset=padded_total_numel,
                                                    size=p.numel(),
                                                    padded_size=padded_numel)
            padded_total_numel += padded_numel
            log_all_ranks_sorted(f"padded_numel: {padded_numel} total_numel: {padded_total_numel}")

        log_all_ranks_sorted(f"param_range_map_global: {param_range_map_global}")

        flat_param_buffer = torch.empty(padded_total_numel, dtype=param_group_dtypes.param_dtype, device=self.device)
        # remap parameters to the param_buffer
        for p in param_group['params']:
            offset = param_range_map_global[p].offset
            size = param_range_map_global[p].size
            p.data = flat_param_buffer[offset:offset + size].view_as(p.data).copy_(p.data)

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
            log_all_ranks_sorted(
                f"padded_shard_size {padded_shard_size} shard_size {shard_size} offset {offset} param_range_map_local: {param_range_map_local[p]}"
            )
            offset += padded_shard_size

        # *Per-rank* flattened grad accumulation buffer
        per_rank_padded_numel = offset
        grad_acc_buffer = torch.zeros(per_rank_padded_numel,
                                      dtype=param_group_dtypes.grad_accum_dtype,
                                      device=self.device)

        if param_group_dtypes.grad_accum_dtype == param_group_dtypes.param_dtype:
            grad_comm_buffer = grad_acc_buffer
        else:
            grad_comm_buffer = torch.zeros(per_rank_padded_numel,
                                           dtype=param_group_dtypes.param_dtype,
                                           device=self.device)

        if param_group_dtypes.optimizer_dtype == param_group_dtypes.param_dtype:
            # This path allows us to directly write reduce-scatter results to shareded gradient used by the optimizer.
            # This case typically happens for:
            # - For Z1: bf16/fp32 training (possibly with torch's autocast)
            # - For Z2/3: bf16/fp32 training (possibly with torch's autocast) + No gradient accumulation training
            grad_for_optimizer = grad_acc_buffer
        else:
            # This path requires an additional buffer and copy.
            # This is necessary for:
            # - Mixed precision training (NVIDIA Apex AMP-style, as DeepSpeed's bf16/fp16 training does)
            # - For Z2/3: Training with gradient accumulation
            grad_for_optimizer = torch.zeros(per_rank_padded_numel,
                                             dtype=param_group_dtypes.optimizer_dtype,
                                             device=self.device)

        param_for_optimizer = torch.empty(per_rank_padded_numel,
                                          dtype=param_group_dtypes.optimizer_dtype,
                                          device=self.device)
        for p in param_group['params']:
            offset_global = param_range_map_global[p].offset
            padded_size_global = param_range_map_global[p].padded_size
            padded_shard_size = padded_size_global // self.world_size
            shard_offset = offset_global + padded_shard_size * self.rank
            log_all_ranks_sorted(
                f"shard_offset: {shard_offset} padded_shard_size: {padded_shard_size} param_range_map_global: {param_range_map_global[p]}"
            )
            copy_src = flat_param_buffer[shard_offset:shard_offset + padded_shard_size]
            log_all_ranks_sorted(
                f"copy_src: {copy_src.shape} {copy_src.dtype} {copy_src.numel()} param_range_map_global: {param_range_map_global[p]}"
            )

            offset_local = param_range_map_local[p].offset
            padded_size_local = param_range_map_local[p].padded_size
            log_all_ranks_sorted(
                f"offset_local: {offset_local} padded_size_local: {padded_size_local} param_range_map_local: {param_range_map_local[p]}"
            )
            param_for_optimizer[offset_local:offset_local + padded_size_local].copy_(copy_src.view(-1))

        return ParamUpdateFlatBuffers(param_buffer=flat_param_buffer,
                                      param_range_map_global=param_range_map_global,
                                      param_range_map_local=param_range_map_local,
                                      grad_acc_buffer=grad_acc_buffer,
                                      grad_comm_buffer=grad_comm_buffer,
                                      grad_for_optimizer=grad_for_optimizer,
                                      param_for_optimizer=param_for_optimizer)

    def _param_group_dtypes(self, param_group: Dict[str, Any]) -> ParamGroupDtypes:
        param_dtype = param_group_dtype(param_group)
        grad_accum_dtype = param_dtype if self.grad_accum_dtype is None else self.grad_accum_dtype
        optimizer_dtype = param_dtype if self.optimizer_dtype is None else self.optimizer_dtype
        return ParamGroupDtypes(param_dtype=param_dtype,
                                grad_accum_dtype=grad_accum_dtype,
                                optimizer_dtype=optimizer_dtype)


class UniversalOptimizer(ABC):

    def __init__(self, optimizer: torch.optim.Optimizer, config: UniversalOptimizerConfig,
                 reduce_bucket_size: int) -> None:

        # for `register_post_accumulate_grad_hook` and `_coalescing_manager`
        assert required_torch_version(min_version=2.1), "UniversalOptimizer requires PyTorch 2.1 or higher."

        self.base_optimizer: torch.optim.Optimizer = optimizer

        self.force_model_dtype: torch.dtype = config.force_model_dtype
        self.zero3_allgather_dtype: torch.dtype = config.zero3_allgather_dtype
        self.reduce_dtype: torch.dtype = config.reduce_dtype
        self.grad_accum_dtype: torch.dtype = config.grad_accum_dtype
        self.optimizer_dtype: torch.dtype = config.optimizer_dtype

        self.is_gradient_accumulation_boundary: bool = True

        self.reduce_op = dist.ReduceOp.AVG

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

        self.gradient_hook_handles = self._create_gradient_handling_hooks()

        self.reduce_tasks: Dict[torch.dtype, List[ReduceTask]] = defaultdict(list)

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

    def _select_optimizer_grad_src(self, task: 'ReduceTask') -> torch.Tensor:
        if task.opt_grad.dtype == task.grad_acc_buf.dtype:
            return task.grad_acc_buf
        if task.opt_grad.dtype == task.recv_buf.dtype:
            return task.recv_buf
        return task.grad_acc_buf

    def gradient_hook(self, param):
        if not self._should_process_gradient(param):
            return

        assert param.dtype in self.comm_buffers, f"Param dtype {param.dtype} not in comm buffers {list(self.comm_buffers.keys())}"
        comm_buffer = self.comm_buffers[param.dtype]

        if comm_buffer.should_flush(param.numel()):
            self.flush_reduce_bucket(param.dtype)

        log_all_ranks_sorted(
            f"gradient_hook param: {id(param)} {param.dtype} {param.numel()} should_flush: {comm_buffer.should_flush(param.numel())}"
        )

        if param.numel() > comm_buffer.get_size():
            # extend buckets
            get_accelerator().current_stream().synchronize()
            comm_buffer.buckets[comm_buffer.current_buffer_idx].reserve(param.numel())

        # reduce_in_buffer = comm_buffer.allocate(param.numel())
        reduce_in_buffer = comm_buffer.allocate(ceil_to_multiple(param.numel(), COMM_PAD_BYTE_SIZE))

        # This ensures the order of reduce_scatter -> copy
        # Without this block, copy may start while reduce_scatter is still running
        comm_buffer.get_event().wait(self.comp_stream)

        copy_src = param.grad.contiguous().view(-1).detach()

        param_buffers = self.param_buffer_map[param]
        recv_buf = param_buffers.grad_comm_buffer
        grad_acc_buf = param_buffers.grad_acc_buffer
        opt_grad = param_buffers.grad_for_optimizer
        self.reduce_tasks[param.dtype].append(
            ReduceTask(param, copy_src, reduce_in_buffer, recv_buf, grad_acc_buf, opt_grad))

        self.rs_comp_done_events[param].record(self.comp_stream)
        self.rs_comp_done_events[param].wait(self.copy_stream)
        with get_accelerator().stream(self.copy_stream):
            reduce_in_buffer.view(-1).narrow(0, 0, copy_src.numel()).copy_(copy_src, non_blocking=True)
            self.rs_copy_done_events[param].record(self.copy_stream)

    def flush_reduce_bucket(self, dtype: torch.dtype):
        if dtype not in self.reduce_tasks:
            return

        log_all_ranks_sorted(f"flush_reduce_bucket dtype: {dtype} #reduce_tasks: {len(self.reduce_tasks[dtype])}")

        self._block_copy_events(dtype)

        with get_coalescing_manager(
                group=None,
                device=self.device,
                async_op=True,
        ) as cm:
            for t in self.reduce_tasks[dtype]:
                log_all_ranks_sorted(
                    f"flush_reduce_bucket t.recv_buf: {t.recv_buf.shape} {t.recv_buf.dtype} {t.recv_buf.numel()} t.send_buf: {t.send_buf.shape} {t.send_buf.dtype} {t.send_buf.numel()}"
                )
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

            for t in self.reduce_tasks[dtype]:
                if t.opt_grad.numel() == 0 or t.opt_grad is t.grad_acc_buf or t.opt_grad is t.recv_buf:
                    continue

                src = self._select_optimizer_grad_src(t)
                t.opt_grad.copy_(src, non_blocking=True)

        for t in self.reduce_tasks[dtype]:
            self.rs_copy_done_events[t.param].record(self.copy_stream)
        self.reduce_tasks[dtype].clear()

        self.comm_buffers[dtype].swap(self.copy_stream)

    def _backward_epilogue(self):
        log_all_ranks_sorted(f"backward_epilogue")
        for dtype in self.comm_buffers.keys():
            self.flush_reduce_bucket(dtype)

    def backward(self, loss, retain_graph=False) -> None:
        loss.backward(retain_graph=retain_graph)
        self._backward_epilogue()

    def step(self, *args, **kwargs):
        self.copy_stream.synchronize()

        original_param_groups = self.base_optimizer.param_groups
        self.base_optimizer.param_groups = self.sharded_param_groups

        self.base_optimizer.step(*args, **kwargs)
        self.base_optimizer.param_groups = original_param_groups

        # Copy updated parameters to the param_buffer if necessary
        issued_cast_copies = False
        with get_accelerator().stream(self.copy_stream):
            for buffers in self.param_update_buffers:
                if buffers.param_for_optimizer.dtype == buffers.param_buffer.dtype:
                    continue

                for p in buffers.param_range_map_global.keys():
                    map_global = buffers.param_range_map_global[p]
                    map_local = buffers.param_range_map_local[p]

                    padded_shard_size = map_local.padded_size
                    if padded_shard_size == 0:
                        continue

                    local_shard_offset = map_global.offset + padded_shard_size * self.rank
                    dst = buffers.param_buffer[local_shard_offset:local_shard_offset + padded_shard_size]
                    src = buffers.param_for_optimizer[map_local.offset:map_local.offset + padded_shard_size]
                    dst.copy_(src, non_blocking=True)
                    issued_cast_copies = True

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

    def zero_grad(self, *args, **kwargs):
        self.base_optimizer.zero_grad(*args, **kwargs)
        self._reset_grad_accum_buffers()

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
            copy_done_event = self.rs_copy_done_events[t.grad]
            copy_done_event.wait(self.comp_stream)

    def _apply_pre_division(self, dtype: torch.dtype):
        if dtype not in self.reduce_tasks:
            return

        for t in self.reduce_tasks[dtype]:
            t.grad.div_(self.world_size)

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

    def _share_grad_and_comm_buffer(self) -> bool:
        return True

    def _should_process_gradient(self, param: torch.Tensor) -> bool:
        return self.is_gradient_accumulation_boundary

    def _accumulate_into_grad_acc_buf(self, task: ReduceTask) -> None:
        if task.grad_acc_buf is task.recv_buf or task.grad_acc_buf.numel() == 0:
            return
        task.grad_acc_buf.copy_(task.recv_buf, non_blocking=True)


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


def configure_universal_optimizer(optimizer: torch.optim.Optimizer, config: UniversalOptimizerConfig,
                                  reduce_bucket_size: int, zero_stage: int):
    if zero_stage == 1:
        optimizer_cls = UniversalOptimizerZ1
    elif zero_stage == 2:
        optimizer_cls = UniversalOptimizerZ2
    else:
        raise ValueError(f"Universal optimizer does not yet support ZeRO stage {zero_stage}.")

    return optimizer_cls(optimizer, config, reduce_bucket_size)
