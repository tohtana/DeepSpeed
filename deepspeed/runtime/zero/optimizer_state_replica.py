# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from copy import deepcopy
from dataclasses import dataclass, replace

import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.adam import FusedAdam

from .partition_parameters import Init
from .stage3 import DeepSpeedZeroOptimizer_Stage3


@dataclass(frozen=True, eq=False)
class Zero3OptimizerSnapshot:
    """A CPU snapshot of one logical rank's ZeRO-3 optimizer shard."""

    rank: int
    step: int
    _layout: tuple
    _state: dict


def _supported_optimizer(engine):
    optimizer = engine.optimizer
    if type(optimizer) is not DeepSpeedZeroOptimizer_Stage3:
        raise ValueError("Optimizer state replication requires ZeRO stage 3.")
    if (optimizer.offload_optimizer or optimizer.offload_param or optimizer.zenflow or engine.has_moe_layers
            or engine.autotp_size() > 1 or engine.mpu is not None or optimizer.sequence_parallel_size != 1):
        raise ValueError("Optimizer state replication requires regular ZeRO-3 data parallelism without offload.")
    if type(optimizer.optimizer) not in (torch.optim.Adam, torch.optim.AdamW, FusedAdam):
        raise ValueError("Optimizer state replication requires Adam, AdamW, or FusedAdam.")
    if any(tensor.dtype != torch.float32 for tensor in optimizer.fp32_partitioned_groups_flat):
        raise ValueError("Optimizer state replication requires FP32 master weights.")
    world_size, rank = dist.get_world_size(), dist.get_rank()
    if any(
            optimizer._get_sub_group_partition_count(i) != world_size
            or optimizer._get_sub_group_partition_rank(i) != rank for i in range(len(optimizer.fp16_groups))):
        raise ValueError("Optimizer state replication requires ZeRO-3 shards across the whole training group.")
    return optimizer


def _layout(optimizer):
    groups = tuple(
        tuple((optimizer.param_names[param], tuple(param.ds_shape), param.dtype, param.partition_numel())
              for param in subgroup) for subgroup in optimizer.fp16_groups)
    group_ids = tuple(optimizer.sub_group_to_group_id.values())
    return (dist.get_world_size(), type(optimizer.optimizer), optimizer.dtype, groups, group_ids)


def _copy_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True, memory_format=torch.contiguous_format)
    return deepcopy(value)


def _broadcast_snapshot(snapshot, source, group):
    sending = dist.get_rank() == source
    metadata = [None]
    if sending:
        leaves, spec = tree_flatten(snapshot._state)
        descriptors = {i: (value.shape, value.dtype) for i, value in enumerate(leaves) if torch.is_tensor(value)}
        values = [None if i in descriptors else value for i, value in enumerate(leaves)]
        metadata[0] = (replace(snapshot, _state={}), spec, values, descriptors)
    dist.broadcast_object_list(metadata, src=source, group=group, device=torch.device("cpu"))
    header, spec, values, descriptors = metadata[0]

    for i, (shape, dtype) in descriptors.items():
        tensor = leaves[i] if sending else torch.empty(shape, dtype=dtype, device="cpu")
        data = tensor.reshape(-1).view(torch.uint8)
        for chunk in data.split(64 * 1024 * 1024):
            if chunk.numel():
                dist.broadcast(chunk, src=source, group=group)
        values[i] = tensor

    return snapshot if sending else replace(header, _state=tree_unflatten(values, spec))


def replicate_optimizer_state(engine, replica_group):
    """Exchange ZeRO-3 optimizer snapshots with one peer over a two-rank Gloo group.

    Both ranks call this after the same completed optimizer step, before beginning the
    next forward or backward. Returns the local snapshot and the peer's snapshot.
    Keep both snapshots in memory until the caller commits a newer step across all pairs.
    """
    optimizer = _supported_optimizer(engine)
    if dist.get_world_size(replica_group) != 2 or dist.get_rank(replica_group) < 0:
        raise ValueError("Optimizer state replication requires membership in a two-rank group.")
    if dist.get_backend(replica_group) != "gloo":
        raise ValueError("Optimizer state replication requires a Gloo replica group.")

    local_snapshot = Zero3OptimizerSnapshot(dist.get_rank(), engine.global_steps, _layout(optimizer),
                                            tree_map(_copy_to_cpu, optimizer.state_dict()))
    for group_rank in range(2):
        source = dist.get_global_rank(replica_group, group_rank)
        snapshot = _broadcast_snapshot(local_snapshot, source, replica_group)
        if source != local_snapshot.rank:
            peer_snapshot = snapshot
    if peer_snapshot.step != local_snapshot.step:
        raise ValueError("Optimizer state replicas must be taken at the same completed step.")
    return local_snapshot, peer_snapshot


def prepare_optimizer_state_recovery():
    """Align parameter IDs across survivors and replacements in a fresh training group.

    Call on every rank after initializing communication and selecting the local device,
    before constructing any recovered ZeRO-3 parameters or engine.
    """
    device = "cpu" if dist.get_backend() == "gloo" else get_accelerator().current_device_name()
    next_id = torch.tensor(Init.param_id, dtype=torch.int64, device=device)
    dist.all_reduce(next_id, op=dist.ReduceOp.MAX)
    Init.param_id = next_id.item()


@torch.no_grad()
def restore_optimizer_state(engine, snapshot):
    """Restore a ZeRO-3 optimizer snapshot on the same logical rank in a fresh engine.

    All training ranks call this with their own snapshot from the same step, after
    rebuilding the original rank layout. The caller restores the rest of its training
    state, including the engine counters, scheduler, RNG, and input position.
    """
    optimizer = _supported_optimizer(engine)
    matches = (isinstance(snapshot, Zero3OptimizerSnapshot) and snapshot.rank == dist.get_rank()
               and snapshot._layout == _layout(optimizer))
    step = snapshot.step if matches else -1
    steps = torch.tensor([step, -step], dtype=torch.int64, device=engine.device)
    dist.all_reduce(steps, op=dist.ReduceOp.MIN, group=optimizer.dp_process_group)
    earliest, neg_latest = steps.tolist()
    if earliest < 0 or earliest != -neg_latest:
        raise ValueError("Every training rank must restore its own matching optimizer snapshot from the same step.")

    state = tree_map(_copy_to_cpu, snapshot._state)
    optimizer.checkpoint_event_prologue()
    optimizer._rigid_load_state_dict(state)
    optimizer.checkpoint_event_epilogue()
