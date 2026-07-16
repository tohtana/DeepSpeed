# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import collections
import threading

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import deepspeed.runtime.zero.parameter_offload as parameter_offload_module
import deepspeed.runtime.zero.partitioned_param_coordinator as coordinator_module
from deepspeed.runtime.base_optimizer import BackwardHookStateManager
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
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

    def __init__(self, ds_id=1, requires_grad=True, is_external_param=False):
        self.ds_id = ds_id
        self.requires_grad = requires_grad
        self.ds_active_sub_modules = set()
        self.ds_persist = False
        self.is_external_param = is_external_param


class FakeModule:

    def __init__(self, ds_id, params):
        self.ds_id = ds_id
        self.params = params


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
    coordinator._PartitionedParameterCoordinator__outer_backward_graph_task_id = None
    coordinator._PartitionedParameterCoordinator__deferred_releases = None
    coordinator._PartitionedParameterCoordinator__deferred_release_lock = None
    return coordinator


def _configure_frozen_boundary_state(offload):
    offload._frozen_boundary_lock = threading.Lock()
    offload._frozen_boundaries = set()


def _configure_release(coordinator, released):
    coordinator._PartitionedParameterCoordinator__trace_mode = ZeRoTraceMode.INVALID
    coordinator.fast_sharding_for_leaf_module = False
    coordinator._PartitionedParameterCoordinator__release_param = \
        lambda param, free_data=True: released.append(param)


def _configure_reset(coordinator, registry):
    coordinator._PartitionedParameterCoordinator__inflight_param_registry = registry
    coordinator._PartitionedParameterCoordinator__trace_mode = ZeRoTraceMode.COMPLETE
    coordinator._PartitionedParameterCoordinator__submodule_order = ()
    coordinator._PartitionedParameterCoordinator__param_order = ()
    coordinator._PartitionedParameterCoordinator__profiler = DummyProfiler()
    coordinator._PartitionedParameterCoordinator__ongoing_fetch_leaf_module_events = {}


def _attach_record(monkeypatch, coordinator, record, module, params, params_to_release=None):
    coordinator._PartitionedParameterCoordinator__trace_mode = ZeRoTraceMode.INVALID
    coordinator.fast_sharding_for_leaf_module = False
    if params_to_release is not None:
        coordinator._PartitionedParameterCoordinator__trace_mode = ZeRoTraceMode.COMPLETE
        coordinator._PartitionedParameterCoordinator__step_id = 1
        coordinator._PartitionedParameterCoordinator__params_to_release = \
            lambda unused_module, unused_step: params_to_release
    monkeypatch.setattr(coordinator_module, "iter_params", lambda unused_module, recurse=False: iter(params))
    monkeypatch.setattr(coordinator_module, "z3_leaf_module", lambda unused_module: False)
    for param in params:
        param.ds_active_sub_modules.add(module.ds_id)
    coordinator.release_sub_module(module, forward=True, deferred_release=record)


def _register_real_multi_grad_hook(coordinator, record, tensors):
    coordinator.set_deferred_release_boundary_count(record, 1)
    return torch.autograd.graph.register_multi_grad_hook(
        tuple(tensors), lambda unused_grads: coordinator.finish_deferred_release(record))


def _multi_grad_handle_removed(handle):
    return all(inner.hooks_dict_ref() is None or inner.id not in inner.hooks_dict_ref() for inner in handle.handles)


def test_grad_bearing_input_retires_locally_before_root(monkeypatch):
    coordinator = _bare_coordinator()
    release_events = []
    coordinator._PartitionedParameterCoordinator__release_param = \
        lambda param, free_data=True: release_events.append(("local", param, set(param.ds_active_sub_modules)))
    frozen = FakeParam(requires_grad=False)
    module = FakeModule(42, [frozen])
    value = torch.randn(4, requires_grad=True)

    coordinator.begin_outer_backward(17)
    record = coordinator.begin_deferred_release()
    handle = _register_real_multi_grad_hook(coordinator, record, (value, ))
    _attach_record(monkeypatch, coordinator, record, module, [frozen])

    (value.sin().sum()).backward()

    assert release_events == [("local", frozen, set())]
    assert coordinator._PartitionedParameterCoordinator__deferred_releases is None
    assert not frozen.ds_active_sub_modules
    handle.remove()

    coordinator.release_outer_backward(17)
    assert len(release_events) == 1


def test_zero_grad_input_uses_root_fallback(monkeypatch):
    coordinator = _bare_coordinator()
    release_events = []
    coordinator._PartitionedParameterCoordinator__release_param = \
        lambda param, free_data=True: release_events.append(("root", param, set(param.ds_active_sub_modules)))
    frozen = FakeParam(requires_grad=False)
    module = FakeModule(42, [frozen])

    coordinator.begin_outer_backward(17)
    record = coordinator.begin_deferred_release()
    _attach_record(monkeypatch, coordinator, record, module, [frozen])

    assert not release_events
    assert frozen.ds_active_sub_modules == {record}

    coordinator.release_outer_backward(17)

    assert release_events == [("root", frozen, set())]
    assert not frozen.ds_active_sub_modules


def test_repeated_shared_invocations_have_unique_local_owners(monkeypatch):
    coordinator = _bare_coordinator()
    release_attempts = []
    coordinator._PartitionedParameterCoordinator__release_param = \
        lambda param, free_data=True: release_attempts.append(set(param.ds_active_sub_modules))
    shared = FakeParam(requires_grad=False)
    module = FakeModule(42, [shared])
    value1 = torch.randn(4, requires_grad=True)
    value2 = torch.randn(4, requires_grad=True)

    coordinator.begin_outer_backward(17)
    records = [coordinator.begin_deferred_release(), coordinator.begin_deferred_release()]
    _register_real_multi_grad_hook(coordinator, records[0], (value1, ))
    _attach_record(monkeypatch, coordinator, records[0], module, [shared])
    _register_real_multi_grad_hook(coordinator, records[1], (value2, ))
    _attach_record(monkeypatch, coordinator, records[1], module, [shared])

    (value1.sin().sum() + value2.cos().sum()).backward()

    assert len(release_attempts) == 2
    assert any(len(owners) == 1 for owners in release_attempts)
    assert release_attempts[-1] == set()
    assert coordinator._PartitionedParameterCoordinator__deferred_releases is None
    coordinator.release_outer_backward(17)
    assert len(release_attempts) == 2


def test_local_shared_release_does_not_consume_no_grad_fallback(monkeypatch):
    coordinator = _bare_coordinator()
    release_attempts = []
    coordinator._PartitionedParameterCoordinator__release_param = \
        lambda param, free_data=True: release_attempts.append(set(param.ds_active_sub_modules))
    shared = FakeParam(requires_grad=False)
    module = FakeModule(42, [shared])
    value = torch.randn(4, requires_grad=True)

    coordinator.begin_outer_backward(17)
    local_record = coordinator.begin_deferred_release()
    fallback_record = coordinator.begin_deferred_release()
    _register_real_multi_grad_hook(coordinator, local_record, (value, ))
    _attach_record(monkeypatch, coordinator, local_record, module, [shared])
    _attach_record(monkeypatch, coordinator, fallback_record, module, [shared])

    value.square().sum().backward()

    assert release_attempts == [{fallback_record}]
    assert shared.ds_active_sub_modules == {fallback_record}

    coordinator.release_outer_backward(17)
    assert release_attempts[-1] == set()
    assert not shared.ds_active_sub_modules


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_recompute_registered_multi_grad_hook_retires_before_root(monkeypatch, use_reentrant):
    coordinator = _bare_coordinator()
    release_events = []
    coordinator._PartitionedParameterCoordinator__release_param = \
        lambda param, free_data=True: release_events.append(("local", set(param.ds_active_sub_modules)))
    frozen = FakeParam(requires_grad=False)
    module = FakeModule(42, [frozen])
    calls = 0

    coordinator.begin_outer_backward(17)

    def checkpointed(value):
        nonlocal calls
        calls += 1
        if calls > 1:
            record = coordinator.begin_deferred_release()
            _register_real_multi_grad_hook(coordinator, record, (value, ))
            _attach_record(monkeypatch, coordinator, record, module, [frozen])
        return value.sin() * value

    value = torch.randn(4, requires_grad=True)
    checkpoint(checkpointed, value, use_reentrant=use_reentrant).sum().backward()

    assert calls > 1
    assert release_events == [("local", set())]
    assert coordinator._PartitionedParameterCoordinator__deferred_releases is None
    coordinator.release_outer_backward(17)
    assert len(release_events) == 1


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_functional_transform_before_frozen_module_retires_locally(monkeypatch, use_reentrant):

    class FrozenLeaf(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4), requires_grad=False)

        def forward(self, value):
            return value * self.weight

    class CheckpointRoot(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.frozen = FrozenLeaf()

        def forward(self, value):
            return checkpoint(lambda inner: self.frozen(torch.sin(inner)), value, use_reentrant=use_reentrant)

    class EpochCoordinator:

        def __init__(self):
            self.active = False
            self.begun = []
            self.finished = []
            self.handles = []

        def is_invalid_trace(self):
            return True

        def has_active_outer_backward(self):
            return self.active

        def begin_deferred_release(self):
            record = object()
            self.begun.append(record)
            return record

        def set_deferred_release_boundary_count(self, unused_record, unused_count):
            pass

        def set_deferred_release_handle(self, record, handle):
            self.handles.append((record, handle))

        def finish_deferred_release(self, record):
            self.finished.append(record)

        def cancel_deferred_release(self, unused_record):
            raise AssertionError("hook registration unexpectedly failed")

    root = CheckpointRoot()
    for module in root.modules():
        module._external_params = {}
        module.ds_external_parameters = lambda: ()
    coordinator = EpochCoordinator()
    attached = []
    offload = object.__new__(DeepSpeedZeRoOffload)
    offload.module = root
    offload.param_coordinator = coordinator
    offload._has_frozen_params = True
    _configure_frozen_boundary_state(offload)
    offload.forward_hooks = []
    offload.backward_hooks = []
    offload.zenflow = False
    offload._begin_outer_backward = lambda: setattr(coordinator, "active", True)
    offload.pre_sub_module_forward_function = lambda unused_module: None
    offload.pre_sub_module_backward_function = lambda unused_module: None
    offload.post_sub_module_backward_function = lambda unused_module: None
    offload.post_sub_module_forward_function = \
        lambda module, deferred_release=None: attached.append((module, deferred_release))
    monkeypatch.setattr(parameter_offload_module, "FWD_MODULE_STACK", [root])

    offload._register_deepspeed_module(root)
    root(torch.randn(4, requires_grad=True)).sum().backward()

    assert len(coordinator.begun) == 1
    assert coordinator.finished == coordinator.begun
    assert any(module is root.frozen and record is coordinator.begun[0] for module, record in attached)
    assert parameter_offload_module.FWD_MODULE_STACK == [root]
    for hook in offload.forward_hooks + offload.backward_hooks:
        hook.remove()


@pytest.mark.parametrize("registration_fails", [False, True])
def test_non_reentrant_early_stop_uses_enclosing_frozen_boundary(monkeypatch, registration_fails):

    class MixedBlock(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(4)
            self.linear = torch.nn.Linear(4, 4)
            for param in self.norm.parameters():
                param.requires_grad_(False)

        def forward(self, value):
            return self.linear(self.norm(value))

    class CheckpointRoot(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.block = MixedBlock()

        def forward(self, value):
            return checkpoint(self.block, value, use_reentrant=False)

    class EpochCoordinator:

        def __init__(self):
            self.active = False
            self.begun = []
            self.finished = []
            self.protect_all = {}

        def is_invalid_trace(self):
            return True

        def has_active_outer_backward(self):
            return self.active

        def begin_deferred_release(self, protect_all_params=False):
            record = object()
            self.begun.append(record)
            self.protect_all[record] = protect_all_params
            return record

        def set_deferred_release_boundary_count(self, unused_record, unused_count):
            pass

        def finish_deferred_release(self, record):
            self.finished.append(record)

        def cancel_deferred_release(self, unused_record):
            raise AssertionError("hook registration unexpectedly failed")

    root = CheckpointRoot()
    for module in root.modules():
        module._external_params = {}
        module.ds_external_parameters = lambda: ()
    coordinator = EpochCoordinator()
    attached = []
    offload = object.__new__(DeepSpeedZeRoOffload)
    offload.module = root
    offload.param_coordinator = coordinator
    offload._has_frozen_params = True
    _configure_frozen_boundary_state(offload)
    offload.forward_hooks = []
    offload.backward_hooks = []
    offload.zenflow = False
    offload._begin_outer_backward = lambda: setattr(coordinator, "active", True)
    offload.pre_sub_module_forward_function = lambda unused_module: None
    offload.pre_sub_module_backward_function = lambda unused_module: None
    offload.post_sub_module_backward_function = lambda unused_module: None
    offload.post_sub_module_forward_function = \
        lambda module, deferred_release=None: attached.append((module, deferred_release))
    monkeypatch.setattr(parameter_offload_module, "FWD_MODULE_STACK", [root])

    offload._register_deepspeed_module(root)
    output = root(torch.randn(2, 4, requires_grad=True)).sum()
    if registration_fails:
        bind_frozen_boundary = offload._bind_frozen_boundary

        def fail_early_stop_registration(*args, protect_all_params=False, **kwargs):
            if protect_all_params:
                raise RuntimeError("early-stop registration failed")
            return bind_frozen_boundary(*args, protect_all_params=protect_all_params, **kwargs)

        offload._bind_frozen_boundary = fail_early_stop_registration
        output.backward()
    else:
        output.backward()

    protected_linear_records = [
        record for module, record in attached
        if module is root.block.linear and record is not None and coordinator.protect_all[record]
    ]
    if registration_fails:
        assert len(protected_linear_records) == 1
        assert protected_linear_records[0] not in coordinator.finished
        offload._clear_frozen_boundaries()
    else:
        assert len(protected_linear_records) == 1
        assert protected_linear_records[0] in coordinator.finished
    assert parameter_offload_module.FWD_MODULE_STACK == [root]
    for hook in offload.forward_hooks + offload.backward_hooks:
        hook.remove()


@pytest.mark.parametrize("outer_reentrant", [False, True])
@pytest.mark.parametrize("inner_reentrant", [False, True])
def test_nested_checkpoint_replays_share_exact_forward_boundary(outer_reentrant, inner_reentrant):

    class EpochCoordinator:

        def __init__(self):
            self.begun = []
            self.cancelled = []
            self.finished = []

        def begin_deferred_release(self):
            record = object()
            self.begun.append(record)
            return record

        def set_deferred_release_boundary_count(self, unused_record, unused_count):
            pass

        def cancel_deferred_release(self, record):
            self.cancelled.append(record)

        def finish_deferred_release(self, record):
            self.finished.append(record)

    coordinator = EpochCoordinator()
    offload = object.__new__(DeepSpeedZeRoOffload)
    offload.param_coordinator = coordinator
    offload._has_frozen_params = True
    _configure_frozen_boundary_state(offload)
    pending_boundaries = []
    in_backward = False
    replay_records = []
    local_records = []
    fallback_records = []

    def frozen_call(value):
        input_tensors = (value, ) if value.requires_grad else ()
        if in_backward:
            record = offload._bind_frozen_boundary(input_tensors, pending_boundaries)
            replay_records.append(record)
            (local_records if input_tensors else fallback_records).append(record)
        else:
            offload._register_frozen_boundary(input_tensors, pending_boundaries)
        return value.cos()

    def inner(value):
        return frozen_call(torch.sin(value))

    def outer(value):
        return checkpoint(inner, value, use_reentrant=inner_reentrant)

    value = torch.randn(4, requires_grad=True)
    output = checkpoint(outer, value, use_reentrant=outer_reentrant)
    assert len(pending_boundaries) == 1
    boundary = pending_boundaries[0]

    in_backward = True
    output.sum().backward()

    assert replay_records == coordinator.begun
    assert not coordinator.cancelled
    assert coordinator.finished == local_records
    assert all(record not in coordinator.finished for record in fallback_records)
    assert not boundary.deferred_releases
    assert [record for pending_boundary in pending_boundaries
            for record in pending_boundary.deferred_releases] == fallback_records
    offload._clear_frozen_boundaries()
    assert not pending_boundaries
    assert not offload._frozen_boundaries


def test_repeated_forward_boundaries_bind_lifo_and_cleanup_unused(monkeypatch):

    class EpochCoordinator:

        def __init__(self):
            self.begun = []
            self.cancelled = []
            self.finished = []

        def begin_deferred_release(self):
            record = object()
            self.begun.append(record)
            return record

        def set_deferred_release_boundary_count(self, unused_record, unused_count):
            pass

        def cancel_deferred_release(self, record):
            self.cancelled.append(record)

        def finish_deferred_release(self, record):
            self.finished.append(record)

    coordinator = EpochCoordinator()
    offload = object.__new__(DeepSpeedZeRoOffload)
    offload.param_coordinator = coordinator
    offload._has_frozen_params = True
    _configure_frozen_boundary_state(offload)
    pending_boundaries = []
    first = torch.randn(4, requires_grad=True)
    second = torch.randn(4, requires_grad=True)
    offload._register_frozen_boundary((first, ), pending_boundaries)
    offload._register_frozen_boundary((second, ), pending_boundaries)
    first_boundary, second_boundary = pending_boundaries
    first_handle = first_boundary.handle
    second_handle = second_boundary.handle
    monkeypatch.setattr(torch._C, "_will_engine_execute_node", lambda node: node in second_boundary.grad_nodes)

    second_records = [
        offload._bind_frozen_boundary((second, ), pending_boundaries),
        offload._bind_frozen_boundary((second, ), pending_boundaries),
    ]
    assert second_boundary.deferred_releases == second_records
    assert not first_boundary.deferred_releases
    second.square().sum().backward()
    assert coordinator.finished == second_records
    assert pending_boundaries == [first_boundary]
    assert _multi_grad_handle_removed(second_handle)

    monkeypatch.setattr(torch._C, "_will_engine_execute_node", lambda node: node in first_boundary.grad_nodes)
    first_record = offload._bind_frozen_boundary((first, ), pending_boundaries)
    assert first_boundary.deferred_releases == [first_record]
    offload._clear_frozen_boundaries()

    assert coordinator.finished == second_records
    assert not pending_boundaries
    assert not offload._frozen_boundaries
    assert _multi_grad_handle_removed(first_handle)


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_unused_later_same_module_boundary_does_not_capture_checkpoint_replay(monkeypatch, use_reentrant):

    class FrozenLeaf(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4), requires_grad=False)

        def forward(self, value):
            return value * self.weight

    class CheckpointThenUnusedRoot(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.frozen = FrozenLeaf()

        def checkpointed(self, value):
            value = self.frozen(torch.sin(value))
            return self.frozen(torch.cos(value))

        def forward(self, value):
            used = checkpoint(self.checkpointed, value, use_reentrant=use_reentrant)
            self.frozen(torch.cos(value))
            return used

    class EpochCoordinator:

        def __init__(self):
            self.active = False
            self.begun = []
            self.boundary_counts = {}
            self.registered_counts = []
            self.finished = []

        def is_invalid_trace(self):
            return True

        def has_active_outer_backward(self):
            return self.active

        def begin_deferred_release(self):
            record = object()
            self.begun.append(record)
            return record

        def set_deferred_release_boundary_count(self, record, count):
            self.boundary_counts[record] = count
            self.registered_counts.append(count)

        def finish_deferred_release(self, record):
            self.boundary_counts[record] -= 1
            if self.boundary_counts[record] == 0:
                self.finished.append(record)

        def cancel_deferred_release(self, unused_record):
            raise AssertionError("boundary binding unexpectedly failed")

    root = CheckpointThenUnusedRoot()
    for module in root.modules():
        module._external_params = {}
        module.ds_external_parameters = lambda: ()
    coordinator = EpochCoordinator()
    attached = []
    offload = object.__new__(DeepSpeedZeRoOffload)
    offload.module = root
    offload.param_coordinator = coordinator
    offload._has_frozen_params = True
    _configure_frozen_boundary_state(offload)
    offload.forward_hooks = []
    offload.backward_hooks = []
    offload.zenflow = False
    offload._begin_outer_backward = lambda: setattr(coordinator, "active", True)
    offload.pre_sub_module_forward_function = lambda unused_module: None
    offload.pre_sub_module_backward_function = lambda unused_module: None
    offload.post_sub_module_backward_function = lambda unused_module: None
    offload.post_sub_module_forward_function = \
        lambda module, deferred_release=None: attached.append((module, deferred_release))
    monkeypatch.setattr(parameter_offload_module, "FWD_MODULE_STACK", [root])

    offload._register_deepspeed_module(root)
    root(torch.randn(4, requires_grad=True)).sum().backward()

    assert len(coordinator.begun) == 2
    assert collections.Counter(coordinator.finished) == collections.Counter(coordinator.begun)
    if not use_reentrant:
        assert coordinator.registered_counts == [2, 2]
    assert all(
        any(module is root.frozen and record is expected for module, record in attached)
        for expected in coordinator.begun)
    assert parameter_offload_module.FWD_MODULE_STACK == [root]
    for hook in offload.forward_hooks + offload.backward_hooks:
        hook.remove()


def test_no_grad_forward_boundary_uses_root_fallback(monkeypatch):
    coordinator = _bare_coordinator()
    released = []
    coordinator._PartitionedParameterCoordinator__release_param = \
        lambda param, free_data=True: released.append(param)
    frozen = FakeParam(requires_grad=False)
    module = FakeModule(42, [frozen])
    offload = object.__new__(DeepSpeedZeRoOffload)
    offload.param_coordinator = coordinator
    offload._has_frozen_params = True
    _configure_frozen_boundary_state(offload)
    pending_boundaries = []

    coordinator.begin_outer_backward(17)
    monkeypatch.setattr(coordinator_module, "current_graph_task_id", lambda: 17)
    offload._register_frozen_boundary((), pending_boundaries)
    boundary = pending_boundaries[0]
    record = offload._bind_frozen_boundary((), pending_boundaries)
    _attach_record(monkeypatch, coordinator, record, module, [frozen])

    assert boundary.handle is None
    assert not boundary.deferred_releases
    assert any(record in candidate.deferred_releases for candidate in offload._frozen_boundaries)
    assert frozen.ds_active_sub_modules == {record}
    coordinator.release_outer_backward(17)
    offload._clear_frozen_boundaries()

    assert released == [frozen]
    assert not frozen.ds_active_sub_modules
    assert not pending_boundaries
    assert not offload._frozen_boundaries


def test_replay_hook_registration_failure_cancels_only_new_record(monkeypatch):

    class EpochCoordinator:

        def __init__(self):
            self.begun = []
            self.cancelled = []

        def begin_deferred_release(self):
            record = object()
            self.begun.append(record)
            return record

        def cancel_deferred_release(self, record):
            self.cancelled.append(record)

    coordinator = EpochCoordinator()
    offload = object.__new__(DeepSpeedZeRoOffload)
    offload.param_coordinator = coordinator
    offload._has_frozen_params = True
    _configure_frozen_boundary_state(offload)
    pending_boundaries = []
    offload._register_frozen_boundary((), pending_boundaries)
    boundary = pending_boundaries[0]
    monkeypatch.setattr(
        parameter_offload_module, "register_multi_grad_hook", lambda unused_tensors, unused_callback:
        (_ for _ in ()).throw(RuntimeError("registration failed")))

    with pytest.raises(RuntimeError, match="registration failed"):
        offload._bind_frozen_boundary((torch.randn(4, requires_grad=True), ), pending_boundaries)

    assert coordinator.cancelled == coordinator.begun
    assert not boundary.deferred_releases
    assert pending_boundaries == [boundary]
    offload._clear_frozen_boundaries()
    assert not pending_boundaries


def test_boundary_registration_failure_leaves_invocation_sentinel_aligned(monkeypatch):

    class FrozenRoot(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4), requires_grad=False)

        def forward(self, value):
            return value * self.weight

    class ActiveCoordinator:

        def __init__(self):
            self.begun = []
            self.cancelled = []

        def is_invalid_trace(self):
            return True

        def has_active_outer_backward(self):
            return False

        def begin_deferred_release(self):
            record = object()
            self.begun.append(record)
            return record

        def set_deferred_release_handle(self, unused_record, unused_handle):
            raise AssertionError("registration should fail first")

        def finish_deferred_release(self, unused_record):
            raise AssertionError("backward did not run")

        def cancel_deferred_release(self, record):
            self.cancelled.append(record)

    root = FrozenRoot()
    root._external_params = {}
    root.ds_external_parameters = lambda: ()
    coordinator = ActiveCoordinator()
    attached = []
    offload = object.__new__(DeepSpeedZeRoOffload)
    offload.module = root
    offload.param_coordinator = coordinator
    offload._has_frozen_params = True
    _configure_frozen_boundary_state(offload)
    offload.forward_hooks = []
    offload.backward_hooks = []
    offload.zenflow = False
    offload._begin_outer_backward = lambda: None
    offload.pre_sub_module_forward_function = lambda unused_module: None
    offload.pre_sub_module_backward_function = lambda unused_module: None
    offload.post_sub_module_backward_function = lambda unused_module: None
    offload.post_sub_module_forward_function = \
        lambda unused_module, deferred_release=None: attached.append(deferred_release)
    monkeypatch.setattr(parameter_offload_module, "FWD_MODULE_STACK", [root])
    monkeypatch.setattr(
        parameter_offload_module, "register_multi_grad_hook", lambda unused_tensors, unused_callback:
        (_ for _ in ()).throw(RuntimeError("registration failed")))
    offload._register_deepspeed_module(root)

    with pytest.raises(RuntimeError, match="registration failed"):
        root(torch.randn(4, requires_grad=True))

    assert not coordinator.begun
    assert not coordinator.cancelled
    assert attached == [None]
    assert not offload._frozen_boundaries
    assert parameter_offload_module.FWD_MODULE_STACK == [root]
    for hook in offload.forward_hooks + offload.backward_hooks:
        hook.remove()


def test_nested_same_module_failure_preserves_outer_boundary_slot(monkeypatch):

    class RepeatedFrozenRoot(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4), requires_grad=False)

        def forward(self, value, nested=True):
            if nested:
                return self(value, nested=False)
            return value * self.weight

    class ActiveCoordinator:

        def __init__(self):
            self.begun = []
            self.cancelled = []
            self.handles = []

        def is_invalid_trace(self):
            return True

        def has_active_outer_backward(self):
            return False

        def begin_deferred_release(self):
            record = object()
            self.begun.append(record)
            return record

        def set_deferred_release_handle(self, record, handle):
            self.handles.append((record, handle))

        def finish_deferred_release(self, record):
            raise AssertionError("backward did not run")

        def cancel_deferred_release(self, record):
            self.cancelled.append(record)

    root = RepeatedFrozenRoot()
    root._external_params = {}
    root.ds_external_parameters = lambda: ()
    coordinator = ActiveCoordinator()
    attached = []
    offload = object.__new__(DeepSpeedZeRoOffload)
    offload.module = root
    offload.param_coordinator = coordinator
    offload._has_frozen_params = True
    _configure_frozen_boundary_state(offload)
    offload.forward_hooks = []
    offload.backward_hooks = []
    offload.zenflow = False
    offload._begin_outer_backward = lambda: None
    offload.pre_sub_module_forward_function = lambda unused_module: None
    offload.pre_sub_module_backward_function = lambda unused_module: None
    offload.post_sub_module_backward_function = lambda unused_module: None
    offload.post_sub_module_forward_function = \
        lambda unused_module, deferred_release=None: attached.append(deferred_release)
    monkeypatch.setattr(parameter_offload_module, "FWD_MODULE_STACK", [root])
    offload._register_deepspeed_module(root)
    pre_hook_calls = 0

    def fail_after_inner_zero_hook(unused_module, unused_inputs):
        nonlocal pre_hook_calls
        pre_hook_calls += 1
        if pre_hook_calls == 2:
            raise RuntimeError("inner pre-hook failed")

    failure_handle = root.register_forward_pre_hook(fail_after_inner_zero_hook)

    with pytest.raises(RuntimeError, match="inner pre-hook failed"):
        root(torch.randn(4, requires_grad=True))

    assert not coordinator.begun
    assert not coordinator.cancelled
    assert attached == [None, None]
    assert len(offload._frozen_boundaries) == 2
    offload._clear_frozen_boundaries()
    assert not offload._frozen_boundaries
    assert parameter_offload_module.FWD_MODULE_STACK == [root]
    failure_handle.remove()
    for hook in offload.forward_hooks + offload.backward_hooks:
        hook.remove()


def test_recompute_promotes_only_releasable_frozen_params(monkeypatch):
    coordinator = _bare_coordinator()
    released = []
    coordinator._PartitionedParameterCoordinator__release_param = \
        lambda param, free_data=True: released.append(param)
    trainable = FakeParam(1)
    frozen = FakeParam(2, requires_grad=False)
    reuse_protected = FakeParam(3, requires_grad=False)
    external = FakeParam(4, requires_grad=False, is_external_param=True)
    params = [trainable, frozen, reuse_protected, external]
    module = FakeModule(42, params)

    coordinator.begin_outer_backward(17)
    record = coordinator.begin_deferred_release()
    _attach_record(monkeypatch, coordinator, record, module, params, params_to_release={1, 2})

    assert released == [trainable]
    assert frozen.ds_active_sub_modules == {record}
    assert not reuse_protected.ds_active_sub_modules
    assert not external.ds_active_sub_modules
    coordinator.release_outer_backward(17)
    assert released == [trainable, frozen]


def test_frozen_model_early_stop_promotes_releasable_trainable_params(monkeypatch):
    coordinator = _bare_coordinator()
    released = []
    coordinator._PartitionedParameterCoordinator__release_param = \
        lambda param, free_data=True: released.append(param)
    trainable = FakeParam(1)
    frozen = FakeParam(2, requires_grad=False)
    module = FakeModule(42, [trainable, frozen])

    coordinator.begin_outer_backward(17)
    record = coordinator.begin_deferred_release(protect_all_params=True)
    _attach_record(monkeypatch, coordinator, record, module, [trainable, frozen], params_to_release={1, 2})

    assert not released
    assert trainable.ds_active_sub_modules == {record}
    assert frozen.ds_active_sub_modules == {record}
    coordinator.set_deferred_release_boundary_count(record, 1)
    coordinator.finish_deferred_release(record)
    assert set(released) == {trainable, frozen}
    assert not trainable.ds_active_sub_modules
    assert not frozen.ds_active_sub_modules


def test_empty_deferred_subset_cancels_record(monkeypatch):
    coordinator = _bare_coordinator()
    trainable = FakeParam(1)
    module = FakeModule(42, [trainable])
    coordinator._PartitionedParameterCoordinator__release_param = lambda param, free_data=True: None

    coordinator.begin_outer_backward(17)
    record = coordinator.begin_deferred_release()
    _attach_record(monkeypatch, coordinator, record, module, [trainable])

    assert not record.active
    assert coordinator._PartitionedParameterCoordinator__deferred_releases is None


def test_root_callback_never_consumes_ordinary_module_owner(monkeypatch):
    coordinator = _bare_coordinator()
    release_attempts = []
    coordinator._PartitionedParameterCoordinator__release_param = \
        lambda param, free_data=True: release_attempts.append(set(param.ds_active_sub_modules))
    frozen = FakeParam(requires_grad=False)
    module = FakeModule(42, [frozen])
    ordinary_owner = 99

    coordinator.begin_outer_backward(17)
    record = coordinator.begin_deferred_release()
    _attach_record(monkeypatch, coordinator, record, module, [frozen])
    # Simulate the backward fetch whose module post-backward hook cannot fire
    # for a no-grad input, plus an unrelated live owner.
    frozen.ds_active_sub_modules.add(module.ds_id)
    frozen.ds_active_sub_modules.add(ordinary_owner)
    coordinator.release_outer_backward(17)

    assert frozen.ds_active_sub_modules == {ordinary_owner}
    assert release_attempts == [{ordinary_owner}]


def test_root_residual_completes_missing_no_grad_module_release(monkeypatch):
    coordinator = _bare_coordinator()
    release_attempts = []
    coordinator._PartitionedParameterCoordinator__release_param = \
        lambda param, free_data=True: release_attempts.append(set(param.ds_active_sub_modules))
    frozen = FakeParam(requires_grad=False)
    module = FakeModule(42, [frozen])

    coordinator.begin_outer_backward(17)
    record = coordinator.begin_deferred_release()
    _attach_record(monkeypatch, coordinator, record, module, [frozen])
    frozen.ds_active_sub_modules.add(module.ds_id)
    coordinator.release_outer_backward(17)

    assert not frozen.ds_active_sub_modules
    assert release_attempts == [set()]


def test_executed_no_grad_module_transfers_fetch_owner_to_root(monkeypatch):
    coordinator = _bare_coordinator()
    released = []
    coordinator._PartitionedParameterCoordinator__release_param = \
        lambda param, free_data=True: released.append(param)
    trainable = FakeParam(1)
    frozen = FakeParam(2, requires_grad=False)
    persistent = FakeParam(3)
    persistent.ds_persist = True
    external = FakeParam(4, is_external_param=True)
    params = [trainable, frozen, persistent, external]
    module = FakeModule(42, params)
    monkeypatch.setattr(coordinator_module, "iter_params", lambda unused_module, recurse=False: iter(params))
    monkeypatch.setattr(coordinator_module, "z3_leaf_module", lambda unused_module: False)
    coordinator.fast_sharding_for_leaf_module = False
    for param in params:
        param.ds_active_sub_modules.add(module.ds_id)

    coordinator.begin_outer_backward(17)
    coordinator.defer_missing_post_backward(module)
    records = coordinator._PartitionedParameterCoordinator__deferred_releases
    assert records is not None and len(records) == 1
    record = next(iter(records))
    assert trainable.ds_active_sub_modules == {record}
    assert frozen.ds_active_sub_modules == {record}
    assert not persistent.ds_active_sub_modules
    assert not external.ds_active_sub_modules

    coordinator.release_outer_backward(17)
    assert len(released) == 2
    assert set(released) == {trainable, frozen}


def test_overlapping_root_graph_tasks_and_missing_epoch_fail_closed():
    coordinator = _bare_coordinator()
    with pytest.raises(RuntimeError, match="without a root backward epoch"):
        coordinator.begin_deferred_release()

    coordinator.begin_outer_backward(17)
    with pytest.raises(RuntimeError, match="overlapping backward"):
        coordinator.begin_outer_backward(23)
    assert coordinator._PartitionedParameterCoordinator__outer_backward_graph_task_id == 17


def test_clean_fully_trainable_paths_create_no_lock_or_record(monkeypatch):
    coordinator = _bare_coordinator()
    registry = CountingRegistry()
    released = []
    _configure_release(coordinator, released)
    coordinator._PartitionedParameterCoordinator__trace_mode = ZeRoTraceMode.COMPLETE
    coordinator._PartitionedParameterCoordinator__step_id = 1
    params = [FakeParam(ds_id) for ds_id in (1, 2)]
    module = FakeModule(42, params)
    iter_params_calls = 0

    def counted_iter_params(unused_module, recurse=False):
        nonlocal iter_params_calls
        iter_params_calls += 1
        return iter(params)

    def reject_tuple_allocation(unused_iterable):
        raise AssertionError("ordinary release must not materialize parameters")

    coordinator._PartitionedParameterCoordinator__params_to_release = \
        lambda unused_module, unused_step: {param.ds_id for param in params}
    monkeypatch.setattr(coordinator_module, "iter_params", counted_iter_params)
    monkeypatch.setattr(coordinator_module, "tuple", reject_tuple_allocation, raising=False)
    monkeypatch.setattr(coordinator_module, "z3_leaf_module", lambda unused_module: False)

    for forward in (True, False):
        for param in params:
            param.ds_active_sub_modules.add(module.ds_id)
        coordinator.release_sub_module(module, forward=forward)
    _configure_reset(coordinator, registry)
    coordinator.reset_step()

    assert coordinator._PartitionedParameterCoordinator__deferred_release_lock is None
    assert coordinator._PartitionedParameterCoordinator__deferred_releases is None
    assert coordinator._PartitionedParameterCoordinator__outer_backward_graph_task_id is None
    assert collections.Counter(released) == collections.Counter({param: 2 for param in params})
    assert iter_params_calls == 2
    assert registry.items_calls == 1


def test_root_reset_drains_exception_residual(monkeypatch):
    coordinator = _bare_coordinator()
    registry = CountingRegistry()
    released = []
    _configure_reset(coordinator, registry)
    coordinator._PartitionedParameterCoordinator__release_param = \
        lambda param, free_data=True: released.append(param)
    frozen = FakeParam(requires_grad=False)
    module = FakeModule(42, [frozen])

    coordinator.begin_outer_backward(17)
    record = coordinator.begin_deferred_release()
    _attach_record(monkeypatch, coordinator, record, module, [frozen])
    _configure_reset(coordinator, registry)
    coordinator.reset_step()

    assert released == [frozen]
    assert not frozen.ds_active_sub_modules
    assert coordinator._PartitionedParameterCoordinator__outer_backward_graph_task_id is None
    assert coordinator._PartitionedParameterCoordinator__deferred_releases is None
    assert registry.items_calls == 1


def test_root_pre_backward_requires_the_managers_outer_callback(monkeypatch):
    offload = object.__new__(DeepSpeedZeRoOffload)
    coordinator = _bare_coordinator()
    offload.param_coordinator = coordinator
    manager = BackwardHookStateManager()
    offload._backward_hook_state_manager = manager
    monkeypatch.setattr(parameter_offload_module, "current_graph_task_id", lambda: 17)

    with pytest.raises(RuntimeError, match="did not register"):
        offload._begin_outer_backward()

    manager.post_backward_callback_queued = True
    manager.post_backward_callback_graph_task_id = 23
    with pytest.raises(RuntimeError, match="did not register"):
        offload._begin_outer_backward()

    manager.post_backward_callback_graph_task_id = 17
    offload._begin_outer_backward()
    assert coordinator._PartitionedParameterCoordinator__outer_backward_graph_task_id == 17


def test_manager_callback_registration_failure_publishes_no_outer_epoch(monkeypatch):
    manager = BackwardHookStateManager()
    manager.enter_backward()

    class FailingExecutionEngine:

        @staticmethod
        def queue_callback(callback):
            raise RuntimeError("registration failed")

    monkeypatch.setattr(torch.autograd.Variable, "_execution_engine", FailingExecutionEngine())
    monkeypatch.setattr(torch._C, "_current_graph_task_id", lambda: 17)

    with pytest.raises(RuntimeError, match="registration failed"):
        manager.queue_post_backward_callback()

    assert not manager.post_backward_callback_queued
    assert manager.post_backward_callback_graph_task_id is None


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_manager_reuses_one_outer_callback_before_checkpoint_recompute(use_reentrant):
    manager = BackwardHookStateManager()
    events = []

    class RootPreBackward(torch.autograd.Function):

        @staticmethod
        def forward(ctx, value):
            return value

        @staticmethod
        def backward(ctx, grad):
            events.append(("root", torch._C._current_graph_task_id()))
            return grad

    calls = 0

    def checkpointed(value):
        nonlocal calls
        calls += 1
        if calls > 1:
            events.append(("recompute", torch._C._current_graph_task_id()))
        return torch.sin(value) * value

    manager.register_grad_acc_post_hook(lambda: events.append(("callback", torch._C._current_graph_task_id())))
    value = torch.randn(4, requires_grad=True)
    output = RootPreBackward.apply(checkpoint(checkpointed, value, use_reentrant=use_reentrant))

    def output_hook(grad):
        events.append(("output", torch._C._current_graph_task_id()))
        manager.enter_backward()
        assert manager.queue_post_backward_callback()
        assert manager.queue_post_backward_callback()
        return grad

    output.register_hook(output_hook)
    output.sum().backward()

    names = [name for name, _ in events]
    assert names[0:2] == ["output", "root"]
    assert names[-1] == "callback"
    assert names.count("callback") == 1
    assert names.index("root") < names.index("recompute")
    assert manager.post_backward_callback_graph_task_id == events[0][1] == events[-1][1]
