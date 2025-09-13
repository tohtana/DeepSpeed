# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass

import torch

import deepspeed.comm as dist
from deepspeed.comm.torch import get_coalescing_manager
from deepspeed.utils.torch import required_torch_version
from deepspeed.accelerator import get_accelerator

from .config import UniversalOptimizerConfig

# https://github.com/NVIDIA/nccl/issues/413#issuecomment-720634194
COMM_PAD_BYTE_SIZE = 128


def ceil_to_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


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
class ParamUpdateBuffers:
    param_buffer: torch.Tensor
    offset_and_sizes: List[Tuple[int, int]]
    grad_for_optimizer: torch.Tensor
    param_for_optimizer: torch.Tensor


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

    def swap(self) -> None:
        raise NotImplementedError("swap is not implemented yet.")


@dataclass
class ReduceTask:
    param: torch.Tensor
    grad: torch.Tensor
    send_buf: torch.Tensor


class ParamUpdateGroupContainer:
    """ A container of ParamUpdateGroup, each group is identified by (dtype, param_group).
    """

    def __init__(self, optimizer: torch.optim.Optimizer, device: torch.device, world_size: int, rank: int,
                 grad_accum_dtype: Optional[torch.dtype], optimizer_dtype: Optional[torch.dtype]) -> None:

        self.optimizer = optimizer

        self.world_size = world_size
        self.rank = rank

        self.device = device
        self.grad_accum_dtype = grad_accum_dtype
        self.optimizer_dtype = optimizer_dtype

        # Initialize buffers for param update for each param group
        for param_group in optimizer.param_groups:
            if len(param_group['params']) == 0:
                continue
            param_group['param_update_buffers'] = self._init_param_update_buffers(param_group)

    def _init_param_update_buffers(self, param_group: Dict[str, Any]) -> None:

        total_numel = sum(p.numel() for p in param_group['params'])

        # Make sure shards are aligned to the pad size
        per_rank_padded_elems, total_padded_elems = sharded_counts_and_totals(total_numel,
                                                                              param_group_dtype(param_group),
                                                                              self.world_size)

        param_offset_and_sizes: List[Tuple[int, int]] = []
        offset = 0
        for p in param_group['params']:
            param_offset_and_sizes.append((offset, p.numel()))
            offset += p.numel()

        param_buffer = torch.empty(total_padded_elems, dtype=param_group_dtype(param_group), device=self.device)
        # remap parameters to the param_buffer
        for p, (offset, size) in zip(param_group['params'], param_offset_and_sizes):
            p.data = param_buffer[offset:offset + size].view_as(p.data)

        grad_acc_buffer = torch.empty(
            per_rank_padded_elems,
            dtype=param_group_dtype(param_group) if self.grad_accum_dtype is None else self.grad_accum_dtype,
            device=self.device)

        if self.optimizer_dtype is None or self.optimizer_dtype == grad_acc_buffer.dtype:
            grad_for_optimizer = grad_acc_buffer
        else:
            grad_for_optimizer = torch.empty(per_rank_padded_elems, dtype=self.optimizer_dtype, device=self.device)

        if self.optimizer_dtype is None or self.optimizer_dtype == param_group_dtype(param_group):
            param_for_optimizer = param_buffer[self.rank * per_rank_padded_elems:(self.rank + 1) *
                                               per_rank_padded_elems]
        else:
            param_for_optimizer = torch.empty(per_rank_padded_elems, dtype=self.optimizer_dtype, device=self.device)

        return ParamUpdateBuffers(param_buffer=param_buffer,
                                  offset_and_sizes=param_offset_and_sizes,
                                  grad_for_optimizer=grad_for_optimizer,
                                  param_for_optimizer=param_for_optimizer)


class UniversalOptimizer:

    def __init__(self, optimizer: torch.optim.Optimizer, config: UniversalOptimizerConfig,
                 reduce_bucket_size: int) -> None:

        # for `register_post_accumulate_grad_hook`
        assert required_torch_version(min_version=2.1), "UniversalOptimizer requires PyTorch 2.1 or higher."

        self.base_optimizer: torch.optim.Optimizer = optimizer

        self.force_model_dtype: torch.dtype = config.force_model_dtype
        self.zero3_allgather_dtype: torch.dtype = config.zero3_allgather_dtype
        self.reduce_dtype: torch.dtype = config.reduce_dtype
        self.grad_accum_dtype: torch.dtype = config.grad_accum_dtype
        self.optimizer_dtype: torch.dtype = config.optimizer_dtype

        self.reduce_op = dist.ReduceOp.AVG

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

        self.param_update_group_container = ParamUpdateGroupContainer(optimizer=self.base_optimizer,
                                                                      grad_accum_dtype=self.grad_accum_dtype,
                                                                      optimizer_dtype=self.optimizer_dtype)

        self._create_gradient_handling_hooks()

        self.reduce_tasks: Dict[torch.dtype, List[ReduceTask]] = defaultdict(list)

        self.comp_stream = get_accelerator().current_stream()
        self.rs_stream = get_accelerator().Stream(priority=-1)
        self.copy_stream = get_accelerator().Stream(priority=-1)

        self.rs_comp_done_events = defaultdict(get_accelerator().Event)
        self.rs_copy_done_events = defaultdict(get_accelerator().Event)

    def gradient_hook(self, param):
        assert param.dtype in self.comm_buffers, f"Param dtype {param.dtype} not in comm buffers {list(self.comm_buffers.keys())}"
        comm_buffer = self.comm_buffers[param.dtype]

        if comm_buffer.should_flush(param.numel()):
            # should swap inside
            comm_buffer.flush(param.dtype)

        if param.numel() > comm_buffer.get_size():
            # extend buckets
            get_accelerator().current_stream().synchronize()
            comm_buffer.buckets[comm_buffer.current_buffer_idx].reserve(param.numel())

        reduce_in_buffer = comm_buffer.allocate(param.numel())

        # This ensures the order of reduce_scatter -> copy
        # Without this block, copy may start while reduce_scatter is still running
        comm_buffer.get_event().wait(self.comp_stream)

        copy_src = param.grad.contiguous().view(-1).detach()

        self.reduce_tasks[param.dtype].append(ReduceTask(param, copy_src, reduce_in_buffer))

        rs_comp_done_event = self.rs_comp_done_events[param].record(self.comp_stream)
        rs_comp_done_event.wait(self.copy_stream)
        with get_accelerator().stream(self.copy_stream):
            reduce_in_buffer.copy_(copy_src, non_blocking=True)
            self.rs_copy_done_events[param].record(self.copy_stream)

    def gradient_hook_epilogue(self):
        for comm_buffer in self.comm_buffers.values():
            comm_buffer.flush()

    def flush_reduce_bucket(self, dtype: torch.dtype):
        if dtype not in self.reduce_tasks:
            return

        self._block_copy_events(dtype)

        with get_coalescing_manager(async_ops=True) as cm:
            for send_buf in [t.send_buf for t in self.reduce_tasks[dtype]]:
                dist.all_reduce(
                    send_buf,
                    op=self.reduce_op,
                    group=None,
                    async_op=True,
                )

        with get_accelerator().stream(self.copy_stream):
            cm.wait()

            for t in self.reduce_tasks[dtype]:
                param = t.grad  # get param from grad tensor
                grad_buf = param.grad.view(-1)

                if grad_buf.numel() == 0:
                    continue

                offset = 0  # get offset from param if necessary
                recv_buf = t.send_buf.view(-1)[offset:offset + grad_buf.numel()]
                grad_buf.copy_(recv_buf, non_blocking=True)

        self.rs_copy_done_events[param].record(self.copy_stream)

        for t in self.reduce_tasks[dtype]:
            self.rs_copy_done_events[t.param].record(self.comp_stream)
        self.reduce_tasks[dtype].clear()

    def backward_epilogue(self):
        # flash all
        for comm_buffer in self.comm_buffers.values():
            comm_buffer.flush()

    def backward(self, loss, retain_graph=False) -> None:
        loss.backward(retain_graph=retain_graph)
        self.backward_epilogue()

    def _create_gradient_handling_hooks(self):
        for i, param_group in enumerate(self.base_optimizer.param_groups):
            for param in param_group:
                if param.requires_grad:
                    return param.register_post_accumulate_grad_hook(self.gradient_hook)

    def _create_comm_buffers(self, reduce_bucket_size: int) -> Dict[torch.dtype, CommDoubleBuffer]:
        dtypes: Set[torch.dtype] = set()
        for param_group in self.base_optimizer.param_groups:
            if len(param_group['params']) == 0:
                continue
            dtypes.add(param_group_dtype(param_group))

        # Use communication buffer per dtype
        comm_buffers: Dict[torch.dtype, CommDoubleBuffer] = {}
        for dtype in dtypes:
            self.comm_buffers[dtype] = CommDoubleBuffer(dtype=dtype,
                                                        buffer_size=reduce_bucket_size,
                                                        device=self.device)
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


def configure_universal_optimizer(optimizer: torch.optim.Optimizer, config: UniversalOptimizerConfig):
    return UniversalOptimizer(optimizer, config)
