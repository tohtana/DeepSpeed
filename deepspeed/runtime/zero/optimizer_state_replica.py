# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass
from typing import Tuple, Union

import torch

from deepspeed import comm as dist
from deepspeed.runtime.utils import view_as_uint8


@dataclass(frozen=True)
class AdamPartitionReplica:
    """One flat ZeRO-3 subgroup's master weights and Adam state on the CPU."""

    fp32_master: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    step: Union[int, torch.Tensor]


@dataclass(frozen=True)
class OptimizerStateReplica:
    """A completed optimizer snapshot from one source rank. Treat its tensors as read-only."""

    source_rank: int
    global_step: int
    subgroups: Tuple[AdamPartitionReplica, ...]


_TRANSFER_CHUNK_ELEMENTS = 4 * 1024 * 1024
_STEP_DTYPES = (torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int64, torch.int32, torch.int16,
                torch.int8, torch.uint8)


class OptimizerStateReplicator:

    def __init__(self, rank, world_size, offset):
        self.source_rank = (rank - offset) % world_size
        self.destination_rank = (rank + offset) % world_size
        self.world_size = world_size

    def _exchange(self, outgoing, incoming):
        input_splits = [0] * self.world_size
        output_splits = [0] * self.world_size
        input_splits[self.destination_rank] = outgoing.size(0)
        output_splits[self.source_rank] = incoming.size(0)
        dist.all_to_all_single(incoming, outgoing, output_split_sizes=output_splits, input_split_sizes=input_splits)

    def _exchange_metadata(self, global_step, states, device):
        metadata = torch.zeros((1 + len(states), 3), dtype=torch.int64, device="cpu")
        engine_step, *optimizer_steps = metadata.unbind()
        engine_step[0] = global_step
        for row, state in zip(optimizer_steps, states):
            dtype_id, ndim, value = row.unbind()
            step = state["step"]
            if torch.is_tensor(step):
                dtype_id.fill_(_STEP_DTYPES.index(step.dtype) + 1)
                ndim.fill_(step.ndim)
                raw_step = view_as_uint8(step.detach().reshape(1))
                view_as_uint8(value)[:raw_step.numel()].copy_(raw_step)
            else:
                value.fill_(step)

        outgoing = metadata.to(device=device)
        incoming = torch.empty_like(outgoing)
        self._exchange(outgoing, incoming)

        engine_step, *optimizer_steps = incoming.cpu().unbind()
        steps = []
        for row in optimizer_steps:
            dtype_id, ndim, value = row.unbind()
            dtype_index = dtype_id.item()
            if dtype_index == 0:
                steps.append(value.item())
            else:
                dtype = _STEP_DTYPES[dtype_index - 1]
                byte_count = torch.empty((), dtype=dtype, device="cpu").element_size()
                step = view_as_uint8(value)[:byte_count].view(dtype).reshape((1, ) * ndim.item())
                steps.append(step.clone())
        return engine_step[0].item(), steps

    def _copy_partition(self, partition, buffer):
        replica = torch.empty_like(partition, device="cpu")
        for start in range(0, partition.numel(), buffer.numel()):
            size = min(buffer.numel(), partition.numel() - start)
            incoming = buffer[:size]
            self._exchange(partition[start:start + size].detach(), incoming)
            replica[start:start + size].copy_(incoming)
        return replica

    def replicate(self, optimizer, global_step):
        masters = optimizer.fp32_partitioned_groups_flat
        states = [optimizer.optimizer.state[master] for master in masters]
        source_step, steps = self._exchange_metadata(global_step, states, masters[0].device)
        buffer_numel = max(1, min(_TRANSFER_CHUNK_ELEMENTS, max(master.numel() for master in masters)))
        buffer = masters[0].new_empty(buffer_numel)
        subgroups = tuple(
            AdamPartitionReplica(self._copy_partition(master, buffer), self._copy_partition(state["exp_avg"], buffer),
                                 self._copy_partition(state["exp_avg_sq"], buffer), step)
            for master, state, step in zip(masters, states, steps))
        return OptimizerStateReplica(self.source_rank, source_step, subgroups)
