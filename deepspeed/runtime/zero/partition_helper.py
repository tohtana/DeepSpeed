# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Sharding planner for ZeRO‑1/2 and native NCCL reduce‑scatter.

This module provides alignment-aware parameter partitioning for efficient NCCL
reduce-scatter operations. It supports both contiguous and chunked partitioning
strategies, with automatic padding to ensure NCCL alignment requirements.

**Key Features:**
- Alignment-aware chunked partitioning for native reduce-scatter
- Equal partition sizes across all ranks (required for NCCL reduce-scatter)
- Automatic tensor padding to meet 4-byte alignment requirements
- Backward compatibility with existing DeepSpeed ZeRO APIs

**Partitioning Modes:**
- `native_reduce_scatter=False`: Traditional equal-sized partitioning (legacy behavior)
- `native_reduce_scatter=True`: Round-robin parameter distribution with alignment support

The following legacy bookkeeping symbols from *DeepSpeed ZeRO stage_1_and_2.py*
now have direct equivalents:

| DeepSpeed variable / helper | New API |
|-----------------------------|----------------------------------------------|
| `partition_size` | `layout.partition_size` |
| `params_in_partition` | `layout.params_in_partition` |
| `params_not_in_partition` | `layout.params_not_in_partition` |
| `first_offset` | `layout.first_offset` |
| `is_param_in_current_partition(pid)` | `layout.is_param_in_partition(param)` |
| `grad_position[pid]` | `layout.grad_position(param)` |
| `param_to_partition_ids[pid]` | `planner.param_to_partition_ids[param]` |
| `total_grads_in_partition` | `layout.total_grads_in_partition` (alias) |
| `grad_partition_insertion_offset[pid]` | `layout.grad_partition_insertion_offset(param)` |
| `grad_start_offset[pid]` | `layout.grad_start_offset(param)` |
| `first_param_index_in_partition` | `layout.first_param_index_in_partition` |

`RankShardLayout` pre‑computes per‑parameter offsets within the *rank‑local
flattened buffer*, so these queries are **O(1)**.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence
import torch

# ---------------------------------------------------------------------------
# Basic dataclasses ----------------------------------------------------------
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardRange:
    """Half‑open interval within one tensor (flattened view)."""

    start: int
    length: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.length < 0:
            raise ValueError("Invalid ShardRange")

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass
class ParamShardSpec:
    """Slice of a parameter that lives on one rank."""

    param: torch.nn.Parameter
    shard_range: ShardRange  # param‑local coordinates
    global_offset: int  # absolute offset in concatenated space. -1 means not set

    # --------------------------------------------------------------
    def view_tensor_slice(self) -> torch.Tensor:
        """Return *view* of the shard (no copy)."""
        return self.param.view(-1)[self.shard_range.start:self.shard_range.end]


# ---------------------------------------------------------------------------
# Per‑rank layout ------------------------------------------------------------
# ---------------------------------------------------------------------------


@dataclass
class RankShardLayout:
    """All information for **one rank** after partitioning."""

    rank: int
    shard_specs: List[ParamShardSpec] = field(default_factory=list)

    # filled by planner -------------------------------------------------
    left_boundary: int = 0  # global start index
    partition_size: int = 0  # logical size incl. padding
    padding: int = 0  # trailing pad elements
    _all_params: Sequence[torch.nn.Parameter] | None = None  # injected

    # derived maps (populated in _finalise) -----------------------------
    _grad_positions: Dict[torch.nn.Parameter, int] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Convenience – size / buffer helpers ------------------------------
    def total_numel(self) -> int:
        """Number of *real* elements owned by this rank."""
        return sum(s.shard_range.length for s in self.shard_specs)

    total_grads_in_partition = total_numel  # DeepSpeed alias

    @property
    def params_in_partition(self) -> List[torch.nn.Parameter]:
        return list(self._grad_positions.keys())

    @property
    def params_not_in_partition(self) -> List[torch.nn.Parameter]:
        if self._all_params is None:
            raise RuntimeError("Planner did not inject _all_params")
        in_set = set(self._grad_positions.keys())
        return [p for p in self._all_params if p not in in_set]

    @property
    def first_offset(self) -> int:
        """Offset within the *first* parameter slice owned by this rank."""
        first_spec = self.shard_specs[0] if self.shard_specs else None
        return first_spec.shard_range.start if first_spec else 0

    # ------------------------------------------------------------------
    # Internal – called by planner ------------------------------------
    def _finalise_offsets(self) -> None:
        """Compute `_grad_positions` (local offsets) in one pass."""
        offset = 0
        for spec in self.shard_specs:
            if spec.param not in self._grad_positions:  # first slice of param in this rank
                self._grad_positions[spec.param] = offset
            offset += spec.shard_range.length


# ---------------------------------------------------------------------------
# Planner -------------------------------------------------------------------
# ---------------------------------------------------------------------------


class PartitionPlanner:
    """Create per‑rank layouts + global maps (e.g. `param_to_partition_ids`)."""

    def __init__(self,
                 params: Sequence[torch.nn.Parameter],
                 world_size: int,
                 *,
                 native_reduce_scatter: bool = False,
                 alignment_factor: int = 1) -> None:
        self.params = list(params)
        self.world_size = world_size
        self.native_reduce_scatter = native_reduce_scatter
        self.alignment_factor = alignment_factor  # NCCL alignment requirement in elements
        self.total_elems = sum(p.numel() for p in self.params)
        self.total_padding = 0  # Will be set during planning

        # build layouts --------------------------------------------------
        self.rank_layouts: List[RankShardLayout] = self._plan()

        # inject param list & compute offsets ---------------------------
        for lay in self.rank_layouts:
            lay._all_params = self.params  # type: ignore[attr-defined]
            lay._finalise_offsets()

        # build param‑>ranks map ----------------------------------------
        self.param_to_partition_ids: Dict[torch.nn.Parameter, List[int]] = {p: [] for p in self.params}
        for rank, lay in enumerate(self.rank_layouts):
            for p in lay.params_in_partition:
                self.param_to_partition_ids[p].append(rank)

    # ------------------------------------------------------------------
    def layout_for_rank(self, rank: int) -> RankShardLayout:
        return self.rank_layouts[rank]

    def get_shard_spec(self, param: torch.nn.Parameter, rank: int) -> ParamShardSpec:
        """Get the shard spec for a parameter on a specific rank.

        Args:
            param: The parameter to find
            rank: The rank to query

        Returns:
            ParamShardSpec for the parameter on the given rank

        Raises:
            ValueError: If parameter is not found in this planner
        """
        layout = self.layout_for_rank(rank)
        for spec in layout.shard_specs:
            if spec.param is param:
                return spec

        # Parameter not found - check if it's in our parameter list at all
        if param not in self.params:
            raise ValueError(f"Parameter not found in planner. This planner contains {len(self.params)} parameters.")
        else:
            raise ValueError(f"Parameter found in planner but has no shard on rank {rank}. "
                             f"This can happen if the rank has no data for this parameter.")

    def get_shard_size(self, param: torch.nn.Parameter, rank: int) -> int:
        """Get the shard size for a parameter on a specific rank.

        Args:
            param: The parameter to find
            rank: The rank to query

        Returns:
            Size of the shard (number of elements) for this parameter on the given rank

        Raises:
            ValueError: If parameter is not found in this planner
        """
        spec = self.get_shard_spec(param, rank)
        return spec.shard_range.length

    def create_data_parallel_partitions(self,
                                        flat_tensor: torch.Tensor,
                                        param_padded_sizes: List[int] | None = None) -> List[torch.Tensor]:
        """Create data parallel partitions from a flat tensor using planner layouts.

        Args:
            flat_tensor: The flattened tensor containing all parameters
            param_padded_sizes: Required when native_reduce_scatter=True.
                              List of padded sizes for each parameter.

        Returns:
            List of tensors, one per rank, containing their partition of the data
        """

        if self.native_reduce_scatter:
            # For native reduce-scatter: use views into pre-padded flat tensor
            if param_padded_sizes is None:
                raise ValueError("param_padded_sizes is required when native_reduce_scatter=True")

            partitions = []

            # Calculate start offsets for each padded parameter
            param_start_offsets = [0]
            for padded_size in param_padded_sizes[:-1]:
                param_start_offsets.append(param_start_offsets[-1] + padded_size)

            for rank in range(self.world_size):
                layout = self.rank_layouts[rank]
                if layout.partition_size == 0:
                    # Empty partition
                    partitions.append(torch.empty(0, dtype=flat_tensor.dtype, device=flat_tensor.device))
                    continue

                # Since parameters are padded to equal shards, we can directly slice
                rank_views = []
                for param_idx, padded_size in enumerate(param_padded_sizes):
                    shard_size = padded_size // self.world_size
                    param_start = param_start_offsets[param_idx]

                    # Get this rank's shard from the padded parameter (view, not copy)
                    shard_start = param_start + rank * shard_size
                    shard_end = shard_start + shard_size
                    rank_views.append(flat_tensor[shard_start:shard_end])

                if rank_views:
                    # Concatenate views - efficient since we're just creating a view
                    rank_partition = torch.cat(rank_views)
                    partitions.append(rank_partition)
                else:
                    partitions.append(
                        torch.zeros(layout.partition_size, dtype=flat_tensor.dtype, device=flat_tensor.device))
        else:
            # For contiguous plan (legacy-style): use contiguous slicing
            partitions = []
            for rank in range(self.world_size):
                layout = self.rank_layouts[rank]
                if layout.partition_size == 0:
                    # Empty partition
                    partitions.append(torch.empty(0, dtype=flat_tensor.dtype, device=flat_tensor.device))
                else:
                    # Extract contiguous partition
                    start = layout.left_boundary
                    end = start + layout.partition_size
                    partition = flat_tensor[start:end].clone()
                    partitions.append(partition)

        return partitions

    # ------------------------------------------------------------------
    # Internal planning algorithms -------------------------------------
    def _plan(self) -> List[RankShardLayout]:
        if self.native_reduce_scatter:
            return self._chunked_plan()
        else:
            return self._contiguous_plan()

    # --------------------------------------------------------------
    def _contiguous_plan(self) -> List[RankShardLayout]:
        # Use legacy partitioning logic: base_size + remainder distribution
        base_size = self.total_elems // self.world_size
        remaining = self.total_elems % self.world_size

        layouts = [RankShardLayout(rank=r) for r in range(self.world_size)]

        # Pre-compute parameter start offsets to avoid repeated calculations
        param_start_offsets = [0]
        for p in self.params[:-1]:
            param_start_offsets.append(param_start_offsets[-1] + p.numel())

        global_off = 0

        for rank in range(self.world_size):
            # Each rank gets base_size + 1 if rank < remaining (same as legacy)
            partition_size = base_size + (1 if rank < remaining else 0)
            rank_start = global_off

            # Fill this rank's partition with parameter data
            current_offset = 0
            remaining_in_partition = partition_size

            for param_idx, p in enumerate(self.params):
                if remaining_in_partition <= 0:
                    break

                param_size = p.numel()
                param_start_global = param_start_offsets[param_idx]

                # Check if this parameter overlaps with current rank's partition
                param_end_global = param_start_global + param_size
                partition_end_global = rank_start + partition_size

                if param_start_global < partition_end_global and param_end_global > rank_start:
                    # Calculate the slice of this parameter that belongs to this rank
                    local_start = max(0, rank_start - param_start_global)
                    local_end = min(param_size, partition_end_global - param_start_global)

                    if local_end > local_start:
                        take = local_end - local_start
                        layouts[rank].shard_specs.append(
                            ParamShardSpec(p, ShardRange(local_start, take), rank_start + current_offset))
                        current_offset += take
                        remaining_in_partition -= take

            self._finalise(layouts[rank], rank_start, partition_size)
            global_off += partition_size

        # Contiguous strategy doesn't need padding
        self.total_padding = 0
        return layouts

    # --------------------------------------------------------------
    def _chunked_plan(self) -> List[RankShardLayout]:
        layouts = [RankShardLayout(rank=r) for r in range(self.world_size)]
        global_off = 0
        total_padding = 0

        for p in self.params:
            n = p.numel()

            # Pad each parameter to be divisible by world_size
            padded_n = ((n + self.world_size - 1) // self.world_size) * self.world_size

            # Further align if needed for NCCL requirements
            if self.alignment_factor > 1:
                shard_size = padded_n // self.world_size
                aligned_shard_size = (
                    (shard_size + self.alignment_factor - 1) // self.alignment_factor) * self.alignment_factor
                padded_n = aligned_shard_size * self.world_size

            # Each rank gets exactly padded_n / world_size elements
            shard_size = padded_n // self.world_size
            param_padding = padded_n - n
            total_padding += param_padding

            # Distribute equal shards to each rank
            for rank in range(self.world_size):
                local_off = rank * shard_size
                # Calculate actual data length for this shard (0 if entirely padding)
                actual_length = max(0, min(shard_size, n - local_off))
                # Use min(local_off, n) to handle padding-only shards
                shard_start = min(local_off, n)
                layouts[rank].shard_specs.append(ParamShardSpec(p, ShardRange(shard_start, actual_length), -1))

            global_off += padded_n

        # Calculate equal partition sizes across all ranks
        total_padded_elems = self.total_elems + total_padding
        partition_size = total_padded_elems // self.world_size

        # Set partition boundaries
        running = 0
        for rank in range(self.world_size):
            self._finalise(layouts[rank], running, partition_size)
            running += partition_size

        # Store total padding needed
        self.total_padding = total_padding

        return layouts

    # --------------------------------------------------------------
    def _finalise(self, layout: RankShardLayout, left: int, size: int) -> None:
        layout.left_boundary = left
        layout.partition_size = size
        layout.padding = max(0, (left + size) - self.total_elems)


class PartitionHelper:
    """Helper to create partition layouts and manage parameter sharding."""

    def __init__(self,
                 native_reduce_scatter: bool = False,
                 alignment_factor: int = 1,
                 gradient_accumulation_dtype: torch.dtype = torch.float32,
                 use_grad_accum_attribute: bool = False,
                 dtype: torch.dtype = torch.float16) -> None:
        self.native_reduce_scatter = native_reduce_scatter
        self.alignment_factor = alignment_factor
        self.gradient_accumulation_dtype = gradient_accumulation_dtype
        self.use_grad_accum_attribute = use_grad_accum_attribute
        self.dtype = dtype  # dtype for reduction operations
        self.planners: List[PartitionPlanner] = []
        self.ranks: List[int] = []  # Store rank for each planner

    def add_planner(self, params: Sequence[torch.nn.Parameter], rank: int, world_size: int) -> None:
        """Create a partition planner for the given parameters."""
        planner = PartitionPlanner(params,
                                   world_size,
                                   native_reduce_scatter=self.native_reduce_scatter,
                                   alignment_factor=self.alignment_factor)
        self.planners.append(planner)
        self.ranks.append(rank)

    def _get_param_group_index(self, param: torch.nn.Parameter) -> int | None:
        """Find which parameter group (planner) a parameter belongs to."""
        for idx, planner in enumerate(self.planners):
            for p in planner.params:
                if p is param:
                    return idx
        return None

    def fill_param_grad_accum_attribute(self, param: torch.nn.Parameter) -> None:
        """Fill gradient accumulation attribute for a parameter.

        This method handles both native_reduce_scatter and legacy modes:
        - For native_reduce_scatter: Creates grad_acc_with_pad with appropriate shard size
        - For legacy mode: Creates/updates grad_accum with accumulated gradients

        Args:
            param: The parameter to process
        """

        if param.grad is not None:
            if self.native_reduce_scatter:
                if not hasattr(param, 'grad_acc_with_pad'):
                    param.grad_acc_with_pad = None
                if param.grad_acc_with_pad is None:
                    # Find which parameter group this param belongs to
                    group_idx = self._get_param_group_index(param)
                    if group_idx is None:
                        raise ValueError(f"Parameter not found in any planner")

                    # Use planner API to get the shard size for this parameter
                    current_rank = self.ranks[group_idx]
                    shard_size = self.planners[group_idx].get_shard_size(param, current_rank)

                    param.grad_acc_with_pad = torch.zeros(shard_size,
                                                          dtype=self.gradient_accumulation_dtype,
                                                          device=param.device)

                # For native_reduce_scatter, also accumulate in grad_accum for flat partition logic
                if param.grad_accum is None:
                    param.grad_accum = param.grad.to(self.gradient_accumulation_dtype)
                else:
                    param.grad_accum.add_(param.grad.to(self.gradient_accumulation_dtype).view(param.grad_accum.shape))
                # Don't clear param.grad for native_reduce_scatter - needed for bucket operations
            else:
                if param.grad_accum is None:
                    param.grad_accum = param.grad.to(self.gradient_accumulation_dtype)
                else:
                    param.grad_accum.add_(param.grad.to(self.gradient_accumulation_dtype).view(param.grad_accum.shape))
                param.grad = None

    def get_gradient_for_reduction(self, param: torch.nn.Parameter) -> torch.Tensor | None:
        """Get the gradient tensor ready for reduction.

        This method handles the logic for retrieving gradients based on whether
        grad_accum_attribute is used and whether native_reduce_scatter is enabled.

        Args:
            param: The parameter to get gradient for

        Returns:
            The gradient tensor ready for reduction, or None if no gradient exists
        """
        if self.use_grad_accum_attribute:
            if param.grad_accum is None:
                return None
            elif self.native_reduce_scatter:
                # For native reduce-scatter, grad_acc_with_pad is already the reduced-size buffer
                return param.grad_acc_with_pad
            else:
                return param.grad_accum.to(self.dtype)
        else:
            return param.grad

    def get_param_gradient_attribute(self, param: torch.nn.Parameter) -> torch.Tensor | None:
        """Get the gradient attribute for a parameter.

        Returns grad_accum if use_grad_accum_attribute is True, otherwise returns grad.
        """
        return param.grad_accum if self.use_grad_accum_attribute else param.grad

    def clear_grad_attribute(self, param: torch.nn.Parameter) -> None:
        """Clear the gradient attribute for a parameter.

        Clears grad_accum if use_grad_accum_attribute is True, otherwise clears grad.
        """
        if self.use_grad_accum_attribute:
            param.grad_accum = None
        else:
            param.grad = None

    def flatten_with_per_param_padding(self,
                                       tensor_list: List[torch.Tensor],
                                       world_size: int,
                                       use_cpu_data: bool = False) -> tuple[torch.Tensor, List[int]]:
        """Flatten tensors with padding after each parameter for native reduce-scatter.

        Each parameter is padded to be divisible by world_size and optionally aligned.
        This allows create_data_parallel_partitions to use views instead of copies.

        Args:
            tensor_list: List of tensors (or params with cpu_data attribute)
            world_size: Number of ranks for partitioning
            use_cpu_data: If True, use param.cpu_data instead of param directly

        Returns:
            Tuple of (flattened_tensor, param_padded_sizes) where param_padded_sizes
            contains the padded size of each parameter
        """
        from torch._utils import _flatten_dense_tensors

        # Extract actual tensors
        tensors = [param.cpu_data for param in tensor_list] if use_cpu_data else tensor_list

        padded_tensors = []
        param_padded_sizes = []

        for tensor in tensors:
            n = tensor.numel()

            # Pad to be divisible by world_size
            padded_n = ((n + world_size - 1) // world_size) * world_size

            # Further align if needed for NCCL requirements
            if self.alignment_factor > 1:
                shard_size = padded_n // world_size
                aligned_shard_size = (
                    (shard_size + self.alignment_factor - 1) // self.alignment_factor) * self.alignment_factor
                padded_n = aligned_shard_size * world_size

            param_padded_sizes.append(padded_n)

            # Add padding if needed
            if padded_n > n:
                padding_size = padded_n - n
                padding = torch.zeros(padding_size, dtype=tensor.dtype, device=tensor.device)
                padded_tensor = torch.cat([tensor.view(-1), padding])
                padded_tensors.append(padded_tensor)
            else:
                padded_tensors.append(tensor.view(-1))

        # Flatten all padded tensors
        flattened = _flatten_dense_tensors(padded_tensors)

        return flattened, param_padded_sizes

    def flatten_without_padding(self,
                                tensor_list: List[torch.Tensor],
                                alignment: int,
                                use_cpu_data: bool = False) -> torch.Tensor:
        """Legacy flatten function with single padding at the end.

        This maintains backward compatibility with the original DeepSpeed behavior
        where padding is only added at the end of all parameters.

        Args:
            tensor_list: List of tensors (or params with cpu_data attribute)
            alignment: Alignment requirement for the total size
            use_cpu_data: If True, use param.cpu_data instead of param directly

        Returns:
            Flattened tensor with padding at the end if needed
        """
        from torch._utils import _flatten_dense_tensors
        from deepspeed.runtime.utils import align_dense_tensors

        # Extract actual tensors
        tensors = [param.cpu_data for param in tensor_list] if use_cpu_data else tensor_list

        # Align and flatten (adds padding at the end if needed)
        aligned_tensors = align_dense_tensors(tensors, alignment)
        flattened = _flatten_dense_tensors(aligned_tensors)

        return flattened
