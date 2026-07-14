# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import collections

from deepspeed.runtime.zero.partitioned_param_coordinator import (PartitionedParameterCoordinator, ZeRoTraceMode)


class CountingLock:

    def __init__(self):
        self.acquisitions = 0

    def __enter__(self):
        self.acquisitions += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeParam:

    def __init__(self):
        self.ds_active_sub_modules = set()
        self.ds_persist = False
        self.is_external_param = False


class CountingRegistry(dict):

    def __init__(self):
        super().__init__()
        self.items_calls = 0

    def items(self):
        self.items_calls += 1
        return super().items()


class DummyProfiler:

    def log_events(self):
        pass

    def reset_events(self):
        pass


def _bare_coordinator():
    coordinator = object.__new__(PartitionedParameterCoordinator)
    coordinator._PartitionedParameterCoordinator__graph_task_leases = {}
    coordinator._PartitionedParameterCoordinator__param_graph_ref_counts = collections.Counter()
    coordinator._PartitionedParameterCoordinator__owner_graph_ref_counts = collections.Counter()
    coordinator._PartitionedParameterCoordinator__graph_cleanup_pending = False
    return coordinator


def test_graph_lifetime_tracking_acquires_lock_once_per_batch():
    coordinator = _bare_coordinator()
    lock = CountingLock()
    coordinator._PartitionedParameterCoordinator__graph_cleanup_lock = lock

    graph_task_id = 17
    lease_type = coordinator._PartitionedParameterCoordinator__GraphTaskLease
    coordinator._PartitionedParameterCoordinator__graph_task_leases[graph_task_id] = lease_type(callback_queued=True)

    fetched_params = {FakeParam() for _ in range(128)}
    coordinator._PartitionedParameterCoordinator__track_graph_task_lifetime(graph_task_id, fetched_params, 42)

    assert lock.acquisitions == 1
    assert all(param.ds_active_sub_modules == {42} for param in fetched_params)
    assert all(count == 1 for count in coordinator._PartitionedParameterCoordinator__param_graph_ref_counts.values())
    assert all(count == 1 for count in coordinator._PartitionedParameterCoordinator__owner_graph_ref_counts.values())

    prefetched_params = {FakeParam() for _ in range(64)}
    coordinator._PartitionedParameterCoordinator__track_graph_task_lifetime(graph_task_id, prefetched_params)

    assert lock.acquisitions == 2
    assert len(coordinator._PartitionedParameterCoordinator__graph_task_leases[graph_task_id].params) == 192


def test_clean_root_reset_skips_graph_cleanup_lock_but_sweeps_inflight_registry():
    coordinator = _bare_coordinator()
    lock = CountingLock()
    registry = CountingRegistry()
    coordinator._PartitionedParameterCoordinator__graph_cleanup_lock = lock
    coordinator._PartitionedParameterCoordinator__inflight_param_registry = registry
    coordinator._PartitionedParameterCoordinator__trace_mode = ZeRoTraceMode.COMPLETE
    coordinator._PartitionedParameterCoordinator__submodule_order = ()
    coordinator._PartitionedParameterCoordinator__param_order = ()
    coordinator._PartitionedParameterCoordinator__profiler = DummyProfiler()
    coordinator._PartitionedParameterCoordinator__ongoing_fetch_leaf_module_events = {}

    coordinator.reset_step()

    assert lock.acquisitions == 0
    assert registry.items_calls == 1


def test_nested_graph_cleanup_preserves_shared_param_and_owner_until_last_lease():
    coordinator = _bare_coordinator()
    coordinator._PartitionedParameterCoordinator__graph_cleanup_lock = CountingLock()
    coordinator._PartitionedParameterCoordinator__inflight_param_registry = {}
    released = []
    coordinator._PartitionedParameterCoordinator__release_param = released.append

    param = FakeParam()
    owner = 42
    lease_type = coordinator._PartitionedParameterCoordinator__GraphTaskLease
    for graph_task_id in (17, 23):
        coordinator._PartitionedParameterCoordinator__graph_task_leases[graph_task_id] = lease_type(
            callback_queued=True)
        coordinator._PartitionedParameterCoordinator__track_graph_task_lifetime(graph_task_id, {param}, owner)

    coordinator.release_unreleased_params(17)

    assert not released
    assert param.ds_active_sub_modules == {owner}
    assert coordinator._PartitionedParameterCoordinator__param_graph_ref_counts[param] == 1
    assert coordinator._PartitionedParameterCoordinator__owner_graph_ref_counts[(param, owner)] == 1

    coordinator.release_unreleased_params(23)

    assert released == [param]
    assert not param.ds_active_sub_modules
    assert not coordinator._PartitionedParameterCoordinator__graph_task_leases
    assert not coordinator._PartitionedParameterCoordinator__param_graph_ref_counts
    assert not coordinator._PartitionedParameterCoordinator__owner_graph_ref_counts
