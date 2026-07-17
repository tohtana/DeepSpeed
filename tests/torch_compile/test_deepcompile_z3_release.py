# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math

import pytest
import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.compile.config import CompileConfig
from deepspeed.compile.util import get_deepcompile_handle, is_deepcompile_supported
from unit.common import DistributedTest

pytestmark = pytest.mark.skipif(not is_deepcompile_supported(),
                                reason="DeepCompile requires CUDA and supported PyTorch")


class TestDeepCompileZ3ReleaseStorage(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    def _device(self):
        return torch.device(get_accelerator().current_device_name())

    def _init_dc(self, pool_budget=1 << 20):
        dc = get_deepcompile_handle()
        dc.init(dist.get_world_group(), CompileConfig(deepcompile=True), 1024)
        if pool_budget is not None:
            dc.set_z3_gather_buffer_pool_budget_for_test(pool_budget)
        return dc

    def _register_param(self, dc, graph_id, ds_id, shape, persistent=False, register_graph=True):
        device = self._device()
        world_size = dist.get_world_size()
        true_numel = math.prod(shape)
        shard_numel = math.ceil(true_numel / world_size)
        rank = dist.get_rank()
        values = torch.arange(rank * shard_numel, (rank + 1) * shard_numel, device=device, dtype=torch.float32)
        grad_buffer = torch.zeros_like(values)
        dc.register_z3_param(ds_id, list(shape), values, grad_buffer, persistent, values.dtype)
        if register_graph:
            dc.register_graph_z3(graph_id, [ds_id])
        return values

    def _gather_view_and_storage(self, shard, graph_id, ds_id):
        gathered = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id)
        gathered = torch.ops.dc.wait_allgather.default(gathered, graph_id, ds_id)
        view = gathered.reshape(-1).narrow(0, 0, gathered.numel() - 1)
        assert view.untyped_storage().data_ptr() == gathered.untyped_storage().data_ptr()
        storage = view.untyped_storage()
        assert storage.nbytes() >= gathered.numel() * gathered.element_size()
        return view, storage

    def _release(self, view, graph_id, ds_id, n_users, synchronize=True):
        torch.ops.dc.release_param.default(view, graph_id, ds_id, n_users)
        if synchronize:
            get_accelerator().synchronize()

    def _expected_view_sum(self, shape):
        world_size = dist.get_world_size()
        shard_numel = math.ceil(math.prod(shape) / world_size)
        values = torch.arange(0, world_size * shard_numel, dtype=torch.float32, device=self._device())
        values = values[:math.prod(shape)].reshape(-1)
        return values.narrow(0, 0, values.numel() - 1).sum()

    def _pool_state(self, dc):
        keys = ("budget", "charged", "high_water", "entries", "checked_out", "retries", "enabled", "initialized")
        return dict(zip(keys, dc.get_z3_gather_buffer_pool_state_for_test()))

    def test_storage_reused_after_release_single_use(self):
        graph_id, ds_id, next_ds_id = 9010, 9011, 9012
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [4097], register_graph=False)
            next_shard = self._register_param(dc, graph_id, next_ds_id, [2049], register_graph=False)
            dc.register_graph_z3(graph_id, [ds_id, next_ds_id])
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            before_ptr = storage.data_ptr()
            self._release(view, graph_id, ds_id, 1)
            assert storage.nbytes() > 0

            next_view, next_storage = self._gather_view_and_storage(next_shard, graph_id, next_ds_id)
            assert next_storage.data_ptr() == before_ptr
            assert torch.allclose(next_view.sum(), self._expected_view_sum([2049]))
            self._release(next_view, graph_id, next_ds_id, 1)
        finally:
            dc.cleanup()

    def test_storage_nonzero_until_final_release_when_multi_use(self):
        graph_id, ds_id = 9020, 9021
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [3])
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            before_release_nbytes = storage.nbytes()
            self._release(view, graph_id, ds_id, 2)
            assert storage.nbytes() == before_release_nbytes
            self._release(view, graph_id, ds_id, 2)
            assert storage.nbytes() == before_release_nbytes
        finally:
            dc.cleanup()

    def test_pool_budget_counts_checked_out_storage(self):
        graph_id = 9050
        first_ds_id, checked_out_ds_id, overlapping_ds_id = 9051, 9052, 9053
        dc = self._init_dc(pool_budget=20_000)
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [4097], register_graph=False)
            checked_out_shard = self._register_param(dc, graph_id, checked_out_ds_id, [2049], register_graph=False)
            overlapping_shard = self._register_param(dc, graph_id, overlapping_ds_id, [1025], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, checked_out_ds_id, overlapping_ds_id])

            first_view, first_storage = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            pool_ptr = first_storage.data_ptr()
            self._release(first_view, graph_id, first_ds_id, 1)

            checked_out_view, checked_out_storage = self._gather_view_and_storage(checked_out_shard, graph_id,
                                                                                  checked_out_ds_id)
            assert checked_out_storage.data_ptr() == pool_ptr

            overlapping_view, overlapping_storage = self._gather_view_and_storage(overlapping_shard, graph_id,
                                                                                  overlapping_ds_id)
            assert overlapping_storage.data_ptr() != pool_ptr
            self._release(overlapping_view, graph_id, overlapping_ds_id, 1)
            assert overlapping_storage.nbytes() == 0

            self._release(checked_out_view, graph_id, checked_out_ds_id, 1)
            assert checked_out_storage.data_ptr() == pool_ptr
            assert checked_out_storage.nbytes() == first_storage.nbytes()
        finally:
            dc.cleanup()

    def test_zero_pool_budget_uses_resize_to_zero_fallback(self):
        graph_id, ds_id = 9060, 9061
        dc = self._init_dc(pool_budget=0)
        try:
            shard = self._register_param(dc, graph_id, ds_id, [4097])
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(view, graph_id, ds_id, 1)
            assert storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_prefetched_storage_is_not_admitted_to_demand_gather_pool(self):
        graph_id, prefetched_ds_id, demand_ds_id = 9070, 9071, 9072
        dc = self._init_dc()
        try:
            prefetched_shard = self._register_param(dc, graph_id, prefetched_ds_id, [4097], register_graph=False)
            demand_shard = self._register_param(dc, graph_id, demand_ds_id, [2049], register_graph=False)
            dc.register_graph_z3(graph_id, [prefetched_ds_id, demand_ds_id])

            torch.ops.dc.prefetch_params_fused.default(graph_id, [prefetched_shard], [prefetched_ds_id])
            prefetched_view, prefetched_storage = self._gather_view_and_storage(prefetched_shard, graph_id,
                                                                                prefetched_ds_id)
            self._release(prefetched_view, graph_id, prefetched_ds_id, 1)
            assert prefetched_storage.nbytes() == 0

            demand_view, demand_storage = self._gather_view_and_storage(demand_shard, graph_id, demand_ds_id)
            self._release(demand_view, graph_id, demand_ds_id, 1)
            assert demand_storage.nbytes() > 0
        finally:
            dc.cleanup()

    def test_prefetch_excludes_existing_pool_storage_from_demand_reuse(self):
        graph_id = 9090
        first_ds_id, prefetched_ds_id, demand_ds_id = 9091, 9092, 9093
        dc = self._init_dc()
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [4097], register_graph=False)
            prefetched_shard = self._register_param(dc, graph_id, prefetched_ds_id, [2049], register_graph=False)
            demand_shard = self._register_param(dc, graph_id, demand_ds_id, [1025], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, prefetched_ds_id, demand_ds_id])

            first_view, first_storage = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            pool_ptr = first_storage.data_ptr()
            self._release(first_view, graph_id, first_ds_id, 1)

            prefetched_view, prefetched_storage = self._gather_view_and_storage(prefetched_shard, graph_id,
                                                                                prefetched_ds_id)
            assert prefetched_storage.data_ptr() == pool_ptr
            dc.set_z3_param_valid_for_test(prefetched_ds_id, False)
            torch.ops.dc.prefetch_params_fused.default(graph_id, [prefetched_shard], [prefetched_ds_id])

            self._release(prefetched_view, graph_id, prefetched_ds_id, 1)
            assert prefetched_storage.nbytes() == 0

            demand_view, demand_storage = self._gather_view_and_storage(demand_shard, graph_id, demand_ds_id)
            self._release(demand_view, graph_id, demand_ds_id, 1)
            assert demand_storage.nbytes() > 0
        finally:
            dc.cleanup()

    def test_prefetch_preparation_failure_rolls_back_storage_exclusion(self):
        graph_id = 9094
        first_ds_id, prefetched_ds_id, demand_ds_id = 9095, 9096, 9097
        dc = self._init_dc()
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [4097], register_graph=False)
            prefetched_shard = self._register_param(dc, graph_id, prefetched_ds_id, [2049], register_graph=False)
            demand_shard = self._register_param(dc, graph_id, demand_ds_id, [1025], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, prefetched_ds_id, demand_ds_id])

            first_view, first_storage = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            pool_ptr = first_storage.data_ptr()
            self._release(first_view, graph_id, first_ds_id, 1)

            prefetched_view, prefetched_storage = self._gather_view_and_storage(prefetched_shard, graph_id,
                                                                                prefetched_ds_id)
            assert prefetched_storage.data_ptr() == pool_ptr
            dc.set_z3_param_valid_for_test(prefetched_ds_id, False)
            dc.set_z3_prefetch_fail_after_exclusions_for_test(1)
            with pytest.raises(RuntimeError, match="injected prefetch preparation failure"):
                torch.ops.dc.prefetch_params_fused.default(graph_id, [prefetched_shard], [prefetched_ds_id])
            dc.set_z3_prefetch_fail_after_exclusions_for_test(0)

            self._release(prefetched_view, graph_id, prefetched_ds_id, 1)
            assert prefetched_storage.nbytes() > 0
            demand_view, demand_storage = self._gather_view_and_storage(demand_shard, graph_id, demand_ds_id)
            assert demand_storage.data_ptr() == pool_ptr
            self._release(demand_view, graph_id, demand_ds_id, 1)
        finally:
            dc.cleanup()

    def test_later_allocator_pressure_preserves_checked_out_working_set(self):
        graph_id = 9100
        first_ds_id, checked_out_ds_id = 9101, 9102
        dc = self._init_dc(pool_budget=None)
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [1_048_577], register_graph=False)
            checked_out_shard = self._register_param(dc, graph_id, checked_out_ds_id, [524_289], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, checked_out_ds_id])

            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            first_view, first_storage = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            self._release(first_view, graph_id, first_ds_id, 1)
            assert first_storage.nbytes() > 2 << 20

            checked_out_view, checked_out_storage = self._gather_view_and_storage(checked_out_shard, graph_id,
                                                                                  checked_out_ds_id)
            assert checked_out_storage.data_ptr() == first_storage.data_ptr()

            before_pressure = self._pool_state(dc)
            assert before_pressure["checked_out"] == 1
            assert before_pressure["charged"] > 2 << 20

            # The free-memory target falls to 2 MiB, but a later retry must not
            # shrink below the byte-exact storage already owned by the pool.
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(2, 8 << 20, 8 * gib)
            after_pressure = self._pool_state(dc)
            assert after_pressure["budget"] == before_pressure["charged"]
            assert after_pressure["charged"] == before_pressure["charged"]
            self._release(checked_out_view, graph_id, checked_out_ds_id, 1)
            assert checked_out_storage.nbytes() == before_pressure["charged"]
        finally:
            dc.cleanup()

    def test_later_allocator_pressure_preserves_idle_non_aligned_working_set(self):
        graph_id, ds_id = 9103, 9104
        dc = self._init_dc(pool_budget=None)
        try:
            shard = self._register_param(dc, graph_id, ds_id, [1_048_577])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(view, graph_id, ds_id, 1)
            assert storage.nbytes() > 2 << 20

            before_pressure = self._pool_state(dc)
            assert before_pressure["charged"] % (2 << 20) != 0

            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(2, 8 << 20, 8 * gib)
            after_pressure = self._pool_state(dc)
            assert after_pressure["budget"] == before_pressure["charged"]
            assert after_pressure["charged"] == before_pressure["charged"]
            assert storage.nbytes() == before_pressure["charged"]

            # Pressure is observed before acquire in production. The candidate
            # must still be available when that imminent acquire searches it.
            next_ds_id = 9105
            next_shard = self._register_param(dc, graph_id, next_ds_id, [524_289], register_graph=False)
            dc.register_graph_z3(graph_id, [ds_id, next_ds_id])
            next_view, next_storage = self._gather_view_and_storage(next_shard, graph_id, next_ds_id)
            assert next_storage.data_ptr() == storage.data_ptr()
            self._release(next_view, graph_id, next_ds_id, 1)
        finally:
            dc.cleanup()

    def test_adaptive_hard_cap_discards_checked_out_storage_on_return(self):
        graph_id, first_ds_id, checked_out_ds_id = 9130, 9131, 9132
        dc = self._init_dc(pool_budget=None)
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [1_048_577], register_graph=False)
            checked_out_shard = self._register_param(dc, graph_id, checked_out_ds_id, [524_289], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, checked_out_ds_id])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            first_view, _ = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            self._release(first_view, graph_id, first_ds_id, 1)
            checked_out_view, checked_out_storage = self._gather_view_and_storage(checked_out_shard, graph_id,
                                                                                  checked_out_ds_id)

            # total / 32 is a 2 MiB hard cap, below this checked-out lease.
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(2, 8 << 20, 64 << 20)
            state = self._pool_state(dc)
            assert state["budget"] == 2 << 20
            assert state["charged"] > state["budget"]
            assert checked_out_storage.nbytes() > state["budget"]
            self._release(checked_out_view, graph_id, checked_out_ds_id, 1)
            assert checked_out_storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_new_demand_evicts_oldest_idle_entry_after_pressure(self):
        graph_id = 9140
        first_ds_id, second_ds_id, demand_ds_id = 9141, 9142, 9143
        dc = self._init_dc(pool_budget=None)
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [524_289], register_graph=False)
            second_shard = self._register_param(dc, graph_id, second_ds_id, [262_145], register_graph=False)
            demand_shard = self._register_param(dc, graph_id, demand_ds_id, [786_433], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, second_ds_id, demand_ds_id])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            first_view, first_storage = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            second_view, second_storage = self._gather_view_and_storage(second_shard, graph_id, second_ds_id)
            self._release(first_view, graph_id, first_ds_id, 1)
            self._release(second_view, graph_id, second_ds_id, 1)

            # Preserve 6 MiB of budget after pressure, then let admission-time
            # LRU reclaim the older first entry for larger new demand.
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(2, 24 << 20, 8 * gib)
            demand_view, _ = self._gather_view_and_storage(demand_shard, graph_id, demand_ds_id)
            self._release(demand_view, graph_id, demand_ds_id, 1)
            assert first_storage.nbytes() == 0
            assert second_storage.nbytes() > 0
        finally:
            dc.cleanup()

    def test_allocator_retry_reset_and_jump_keep_budget_state_consistent(self):
        dc = self._init_dc(pool_budget=None)
        try:
            # Registering a graph keeps the process-global weak pool alive
            # between the individual pressure-seam calls below.
            self._register_param(dc, 9150, 9151, [3])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(10, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(11, gib, 8 * gib)
            assert self._pool_state(dc)["budget"] == 256 << 20

            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, 8 << 20, 8 * gib)
            reset_state = self._pool_state(dc)
            assert reset_state["retries"] == 0
            assert reset_state["budget"] == 256 << 20

            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(5, 16 << 20, 8 * gib)
            jump_state = self._pool_state(dc)
            assert jump_state["retries"] == 5
            assert jump_state["budget"] == 4 << 20
        finally:
            dc.cleanup()

    def test_budget_lowering_discards_checked_out_storage_on_return(self):
        graph_id, first_ds_id, checked_out_ds_id = 9110, 9111, 9112
        dc = self._init_dc(pool_budget=20_000)
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [4097], register_graph=False)
            checked_out_shard = self._register_param(dc, graph_id, checked_out_ds_id, [2049], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, checked_out_ds_id])

            first_view, first_storage = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            self._release(first_view, graph_id, first_ds_id, 1)
            checked_out_view, checked_out_storage = self._gather_view_and_storage(checked_out_shard, graph_id,
                                                                                  checked_out_ds_id)
            assert checked_out_storage.data_ptr() == first_storage.data_ptr()

            dc.set_z3_gather_buffer_pool_budget_for_test(0)
            self._release(checked_out_view, graph_id, checked_out_ds_id, 1)
            assert checked_out_storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_cleanup_clears_process_global_test_override(self):
        first_graph_id, first_ds_id = 9120, 9121
        dc = self._init_dc()
        first_shard = self._register_param(dc, first_graph_id, first_ds_id, [4097])
        first_view, first_storage = self._gather_view_and_storage(first_shard, first_graph_id, first_ds_id)
        self._release(first_view, first_graph_id, first_ds_id, 1)
        assert first_storage.nbytes() > 0
        dc.cleanup()

        second_graph_id, second_ds_id = 9122, 9123
        dc = self._init_dc(pool_budget=None)
        try:
            second_shard = self._register_param(dc, second_graph_id, second_ds_id, [4097])
            second_view, second_storage = self._gather_view_and_storage(second_shard, second_graph_id, second_ds_id)
            self._release(second_view, second_graph_id, second_ds_id, 1)
            assert second_storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_cleanup_is_idempotent_and_releases_registry_state(self):
        graph_id, ds_id = 9160, 9161
        dc = self._init_dc()
        first_shard = self._register_param(dc, graph_id, ds_id, [4097])
        first_view, _ = self._gather_view_and_storage(first_shard, graph_id, ds_id)
        self._release(first_view, graph_id, ds_id, 1)

        dc.cleanup()
        dc.cleanup()

        dc = self._init_dc()
        try:
            second_shard = self._register_param(dc, graph_id, ds_id, [2049])
            second_view, _ = self._gather_view_and_storage(second_shard, graph_id, ds_id)
            assert torch.allclose(second_view.sum(), self._expected_view_sum([2049]))
            self._release(second_view, graph_id, ds_id, 1)
        finally:
            dc.cleanup()

    def test_profile_invalidation_discards_checked_out_pool_storage_immediately(self):
        graph_id = 9080
        first_ds_id, invalidated_ds_id, next_ds_id, reused_ds_id = 9081, 9082, 9083, 9084
        dc = self._init_dc()
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [4097], register_graph=False)
            invalidated_shard = self._register_param(dc, graph_id, invalidated_ds_id, [2049], register_graph=False)
            next_shard = self._register_param(dc, graph_id, next_ds_id, [1025], register_graph=False)
            reused_shard = self._register_param(dc, graph_id, reused_ds_id, [513], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, invalidated_ds_id, next_ds_id, reused_ds_id])

            first_view, first_storage = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            pool_ptr = first_storage.data_ptr()
            self._release(first_view, graph_id, first_ds_id, 1)

            invalidated_view, invalidated_storage = self._gather_view_and_storage(invalidated_shard, graph_id,
                                                                                  invalidated_ds_id)
            assert invalidated_storage.data_ptr() == pool_ptr
            dc.invalidate_gathered_param(invalidated_ds_id)

            # The invalidated view deliberately remains alive here. A new gather must
            # not acquire the storage that profiling removed from pool ownership.
            next_view, next_storage = self._gather_view_and_storage(next_shard, graph_id, next_ds_id)
            assert next_storage.data_ptr() != pool_ptr

            self._release(next_view, graph_id, next_ds_id, 1)
            dc.clear_all_gathered_params()
            del invalidated_view
            get_accelerator().synchronize()

            reused_view, reused_storage = self._gather_view_and_storage(reused_shard, graph_id, reused_ds_id)
            assert reused_storage.data_ptr() == next_storage.data_ptr()
            self._release(reused_view, graph_id, reused_ds_id, 1)
        finally:
            dc.cleanup()

    def test_persistent_param_storage_unchanged_across_release(self):
        graph_id, ds_id = 9030, 9031
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [4], persistent=True)
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            before_ptr = storage.data_ptr()
            before_nbytes = storage.nbytes()
            self._release(view, graph_id, ds_id, 1)
            assert storage.data_ptr() == before_ptr
            assert storage.nbytes() == before_nbytes
        finally:
            dc.cleanup()

    def test_consumer_stream_can_finish_before_storage_reuse(self):
        graph_id, ds_id, next_ds_id = 9040, 9041, 9042
        if not hasattr(torch.cuda, "_sleep"):  #ignore-cuda
            pytest.skip("CUDA sleep helper is unavailable")
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [4097], register_graph=False)
            next_shard = self._register_param(dc, graph_id, next_ds_id, [2049], register_graph=False)
            dc.register_graph_z3(graph_id, [ds_id, next_ds_id])
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            before_ptr = storage.data_ptr()
            result = torch.empty((), device=self._device(), dtype=view.dtype)
            consumer_stream = get_accelerator().Stream()
            with get_accelerator().stream(consumer_stream):
                torch.cuda._sleep(int(1e8))  #ignore-cuda
                result.copy_(view.sum())
                self._release(view, graph_id, ds_id, 1, synchronize=False)

            next_view, next_storage = self._gather_view_and_storage(next_shard, graph_id, next_ds_id)
            get_accelerator().synchronize()
            assert torch.allclose(result, self._expected_view_sum([4097]))
            assert next_storage.data_ptr() == before_ptr
            assert torch.allclose(next_view.sum(), self._expected_view_sum([2049]))
            assert storage.nbytes() > 0
            self._release(next_view, graph_id, next_ds_id, 1)
        finally:
            dc.cleanup()
