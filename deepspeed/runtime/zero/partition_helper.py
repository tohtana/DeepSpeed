# zero_param_partition.py (v5 – DeepSpeed‑compat helpers)
"""
Sharding planner for ZeRO‑1/2 and native NCCL reduce‑scatter.

**v5: Full DeepSpeed compatibility helpers**
-------------------------------------------
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
        if self.start < 0 or self.length <= 0:
            raise ValueError("Invalid ShardRange")

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass
class ParamShardSpec:
    """Slice of a parameter that lives on one rank."""

    param: torch.nn.Parameter
    shard_range: ShardRange  # param‑local coordinates
    global_offset: int       # absolute offset in concatenated space

    # --------------------------------------------------------------
    def view_tensor_slice(self) -> torch.Tensor:
        """Return *view* of the shard (no copy)."""
        return self.param.view(-1)[self.shard_range.start : self.shard_range.end]


# ---------------------------------------------------------------------------
# Per‑rank layout ------------------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class RankShardLayout:
    """All information for **one rank** after partitioning."""

    rank: int
    shard_specs: List[ParamShardSpec] = field(default_factory=list)

    # filled by planner -------------------------------------------------
    left_boundary: int = 0       # global start index
    partition_size: int = 0      # logical size incl. padding
    padding: int = 0             # trailing pad elements
    _all_params: Sequence[torch.nn.Parameter] | None = None  # injected

    # derived maps (populated in _finalise) -----------------------------
    _grad_positions: Dict[torch.nn.Parameter, int] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Convenience – size / buffer helpers ------------------------------
    def total_numel(self) -> int:
        """Number of *real* elements owned by this rank."""
        return sum(s.shard_range.length for s in self.shard_specs)

    total_grads_in_partition = total_numel  # DeepSpeed alias

    def alloc_recv_buffer(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Allocate buffer (logical `partition_size`, incl. padding)."""
        return torch.empty(self.partition_size, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Weight / gradient gathering --------------------------------------
    def gather_weights(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Concatenate all local shards into a flat tensor on `device`/`dtype`."""
        buf = torch.empty(self.total_numel(), dtype=dtype, device=device)
        offset = 0
        for spec in self.shard_specs:
            shard = spec.view_tensor_slice().to(device=device, dtype=dtype, non_blocking=True)
            buf[offset : offset + spec.shard_range.length].copy_(shard)
            offset += spec.shard_range.length
        return buf

    # ------------------------------------------------------------------
    # DeepSpeed‑style query helpers ------------------------------------
    def is_param_in_partition(self, param: torch.nn.Parameter) -> bool:
        return param in self._grad_positions

    # alias for naming parity
    is_param_in_current_partition = is_param_in_partition  # type: ignore

    def grad_position(self, param: torch.nn.Parameter) -> int:
        """Offset **within rank‑local flat buffer** where this param’s first element sits."""
        return self._grad_positions[param]

    grad_partition_insertion_offset = grad_position  # alias
    grad_start_offset = grad_position               # alias

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

    @property
    def first_param_index_in_partition(self) -> int:
        """Index (into `planner.params`) of the first parameter present on this rank."""
        if not self.shard_specs:
            return -1
        param0 = self.shard_specs[0].param
        return self._all_params.index(param0) if self._all_params else -1

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

    def __init__(self, params: Sequence[torch.nn.Parameter], world_size: int, *, strategy: str = "contiguous") -> None:
        self.params = list(params)
        self.world_size = world_size
        self.strategy = strategy
        self.total_elems = sum(p.numel() for p in self.params)

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
    
    def create_data_parallel_partitions(self, flat_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Create data parallel partitions from a flat tensor using planner layouts."""
        partitions = []
        for rank in range(self.world_size):
            layout = self.rank_layouts[rank]
            if layout.total_numel() == 0:
                # Empty partition - create a zero tensor
                partitions.append(torch.empty(0, dtype=flat_tensor.dtype, device=flat_tensor.device))
            else:
                # Extract the partition from flat tensor using layout boundaries
                start = layout.left_boundary
                end = start + layout.total_numel()
                partitions.append(flat_tensor[start:end])
        return partitions

    # ------------------------------------------------------------------
    # Internal planning algorithms -------------------------------------
    def _plan(self) -> List[RankShardLayout]:
        if self.strategy == "contiguous":
            return self._contiguous_plan()
        if self.strategy in {"chunked", "strided"}:
            return self._chunked_plan()
        raise ValueError(f"Unknown strategy: {self.strategy}")

    # --------------------------------------------------------------
    def _contiguous_plan(self) -> List[RankShardLayout]:
        shard_size = (self.total_elems + self.world_size - 1) // self.world_size
        layouts = [RankShardLayout(rank=r) for r in range(self.world_size)]

        current = 0
        used = 0
        global_off = 0
        rank_start = 0

        for p in self.params:
            rem = p.numel()
            local_off = 0
            while rem > 0:
                room = shard_size - used
                take = min(room, rem)
                layouts[current].shard_specs.append(
                    ParamShardSpec(p, ShardRange(local_off, take), global_off)
                )
                rem -= take
                local_off += take
                global_off += take
                used += take
                if used == shard_size and current < self.world_size - 1:
                    self._finalise(layouts[current], rank_start, shard_size)
                    current += 1
                    rank_start = global_off
                    used = 0

        self._finalise(layouts[current], rank_start, shard_size)
        for r in range(current + 1, self.world_size):
            self._finalise(layouts[r], global_off, shard_size)
        return layouts

    # --------------------------------------------------------------
    def _chunked_plan(self) -> List[RankShardLayout]:
        layouts = [RankShardLayout(rank=r) for r in range(self.world_size)]
        global_off = 0
        for p in self.params:
            n = p.numel()
            base = n // self.world_size
            extra = n % self.world_size
            local_off = 0
            for rank in range(self.world_size):
                length = base + (1 if rank < extra else 0)
                if length:
                    layouts[rank].shard_specs.append(
                        ParamShardSpec(p, ShardRange(local_off, length), global_off + local_off)
                    )
                local_off += length
            global_off += n

        running = 0
        for lay in layouts:
            self._finalise(lay, running, lay.total_numel())
            running += lay.total_numel()
        return layouts

    # --------------------------------------------------------------
    def _finalise(self, layout: RankShardLayout, left: int, size: int) -> None:
        layout.left_boundary = left
        layout.partition_size = size
        layout.padding = max(0, (left + size) - self.total_elems)


# ---------------------------------------------------------------------------
# Quick demo -----------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    world = 4
    params = [torch.arange(n) for n in (10, 7, 5)]
    planner = PartitionPlanner(params, world, strategy="contiguous")

    rank = 2
    lay = planner.layout_for_rank(rank)
    p0 = params[0]

    print("rank", rank)
    print("is_param_in_partition(p0)", lay.is_param_in_partition(p0))
    print("grad_position(p0)", lay.grad_position(p0))
    print("param_to_partition_ids[p0]", planner.param_to_partition_ids[p0])
    print("first_param_index_in_partition", lay.first_param_index_in_partition)
