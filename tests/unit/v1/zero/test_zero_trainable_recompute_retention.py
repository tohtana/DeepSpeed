# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from contextlib import contextmanager

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils import set_z3_leaf_modules, z3_leaf_module
from deepspeed.utils.tensor_fragment import safe_get_full_grad

from unit.common import DistributedTest
from unit.v1.zero.test_zero_activation_checkpoint_lifecycle import (
    _EarlyStopCheckpointModel,
    _IncompleteBackwardModel,
    _NoGradInputModel,
    _RecomputeExceptionModel,
    _RecursiveCheckpointModel,
)
from unit.v1.zero.test_zero_user_backward import (
    MultiTensorLeafBlock,
    MultiTensorLeafFrozenModel,
    get_config_dict,
    initialize_distributed,
)

_ACTIVE_BACKWARD = "_PartitionedParameterCoordinator__active_backward_submodules"
_ALL_GATHER = "_PartitionedParameterCoordinator__all_gather_params"
_RETAINED_RECOMPUTE = "_PartitionedParameterCoordinator__retained_recompute_submodules"


def _zero3_config(*, retain, persistence_threshold=0, gradient_accumulation_steps=1):
    config = get_config_dict(3,
                             gradient_accumulation_steps=gradient_accumulation_steps,
                             force_fp32=True,
                             param_persistence_threshold=persistence_threshold)
    zero = config["zero_optimization"]
    zero["stage3_prefetch_bucket_size"] = 0
    zero["stage3_max_reuse_distance"] = 0
    zero["stage3_retain_trainable_params_for_recompute"] = retain
    return config


def _initialize(model, *, retain, persistence_threshold=0, gradient_accumulation_steps=1, leaf_types=None):
    if leaf_types:
        set_z3_leaf_modules(model, leaf_types)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    engine, _, _, _ = deepspeed.initialize(config=_zero3_config(
        retain=retain,
        persistence_threshold=persistence_threshold,
        gradient_accumulation_steps=gradient_accumulation_steps),
                                           model=model,
                                           model_parameters=trainable)
    return engine


def _synchronize():
    get_accelerator().synchronize()
    dist.barrier()


def _assert_clean(engine, *, require_partitioned=True):
    coordinator = engine.optimizer.parameter_offload.get_param_coordinator()
    assert not getattr(coordinator, _ACTIVE_BACKWARD)
    assert not getattr(coordinator, _RETAINED_RECOMPUTE)

    for module_name, module in engine.module.named_modules():
        recompute = getattr(module, "ds_recompute_parameters", set())
        assert not recompute, (f"module {module_name or '<root>'} kept recompute parameters: "
                               f"{[parameter.ds_id for parameter in recompute]}")

    for parameter_name, parameter in engine.module.named_parameters():
        assert not parameter.ds_active_sub_modules, (
            f"parameter {parameter_name} kept active owners: {sorted(parameter.ds_active_sub_modules)}")
        if require_partitioned and not parameter.ds_persist and not parameter.is_external_param:
            assert parameter.ds_status == ZeroParamStatus.NOT_AVAILABLE, (
                f"parameter {parameter_name} stayed resident: {parameter.ds_status}")


def _assert_invocation_state_reset(engine, reset_modules=()):
    parameter_offload = engine.optimizer.parameter_offload
    assert not getattr(parameter_offload, "_recompute_grads_remaining_modules", set())
    module_names = {module: name for name, module in engine.module.named_modules()}
    for module_name, module in engine.module.named_modules():
        assert module.__dict__.get("ds_grads_remaining_graph_task_id",
                                   -1) == -1, (f"module {module_name or '<root>'} kept a recompute graph task")
    for module in reset_modules:
        module_name = module_names[module]
        assert module.__dict__.get("ds_grads_remaining",
                                   0) == 0, (f"module {module_name or '<root>'} kept recompute invocations")


def _collect_gradients(engine):
    gradients = {}
    for name, parameter in engine.module.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = safe_get_full_grad(parameter)
        if gradient is not None:
            gradients[name] = gradient.detach().float().cpu()
    return gradients


class _CheckpointRetentionModel(torch.nn.Module):

    def __init__(self, hidden_dim, use_reentrant):
        super().__init__()
        self.use_reentrant = use_reentrant
        self.first = torch.nn.Linear(hidden_dim, hidden_dim)
        self.second = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)

    def _checkpointed(self, value):
        return self.second(torch.tanh(self.first(value)))

    def forward(self, value):
        value = checkpoint(self._checkpointed, value, use_reentrant=self.use_reentrant)
        return self.head(torch.sin(value))


class _PlainRetentionModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.first = torch.nn.Linear(hidden_dim, hidden_dim)
        self.second = torch.nn.Linear(hidden_dim, 1)

    def forward(self, value):
        return self.second(torch.tanh(self.first(value)))


class _NestedCheckpointRetentionModel(torch.nn.Module):

    def __init__(self, hidden_dim, use_reentrant):
        super().__init__()
        self.use_reentrant = use_reentrant
        self.inner = torch.nn.Linear(hidden_dim, hidden_dim)
        self.outer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)

    def _inner_checkpointed(self, value):
        return torch.tanh(self.inner(value))

    def _outer_checkpointed(self, value):
        value = checkpoint(self._inner_checkpointed, value, use_reentrant=self.use_reentrant)
        return torch.sin(self.outer(value))

    def forward(self, value):
        value = checkpoint(self._outer_checkpointed, value, use_reentrant=self.use_reentrant)
        return self.head(value)


class _RepeatedModuleCheckpointRetentionModel(torch.nn.Module):

    def __init__(self, hidden_dim, use_reentrant):
        super().__init__()
        self.use_reentrant = use_reentrant
        self.shared = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)

    def _checkpointed(self, value):
        value = torch.tanh(self.shared(value))
        return torch.sin(self.shared(value))

    def forward(self, value):
        value = checkpoint(self._checkpointed, value, use_reentrant=self.use_reentrant)
        return self.head(value)


@contextmanager
def _record_target_lifecycle(engine, target):
    coordinator = engine.optimizer.parameter_offload.get_param_coordinator()
    original_gather = getattr(coordinator, _ALL_GATHER)
    original_release = coordinator.release_sub_module
    events = []

    def record_gather(parameters, forward):
        if target in parameters and target.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            events.append({
                "kind": "gather",
                "forward": forward,
                "in_recompute": torch._C._current_graph_task_id() != -1,
                "owners": set(target.ds_active_sub_modules),
            })
        return original_gather(parameters, forward)

    def record_release(submodule, forward=False):
        module_parameters = set(submodule.parameters(recurse=z3_leaf_module(submodule)))
        recompute_parameters = getattr(submodule, "ds_recompute_parameters", set())
        retained_modules = getattr(coordinator, _RETAINED_RECOMPUTE)
        in_recompute = torch._C._current_graph_task_id() != -1
        is_recompute_forward = forward and in_recompute and target in module_parameters
        is_matching_backward = not forward and target in recompute_parameters
        before = {
            "status": target.ds_status,
            "owners": set(target.ds_active_sub_modules),
            "owner_module": submodule.ds_id,
            "retained_modules": set(retained_modules),
        }
        result = original_release(submodule, forward)
        if is_recompute_forward or is_matching_backward:
            events.append({
                "kind": "recompute_forward_release" if is_recompute_forward else "matching_backward_release",
                "before": before,
                "after": {
                    "status": target.ds_status,
                    "owners": set(target.ds_active_sub_modules),
                    "retained_modules": set(retained_modules),
                },
            })
        return result

    setattr(coordinator, _ALL_GATHER, record_gather)
    coordinator.release_sub_module = record_release
    try:
        yield events
    finally:
        setattr(coordinator, _ALL_GATHER, original_gather)
        coordinator.release_sub_module = original_release


@pytest.mark.parametrize("retain", [False, True], ids=["off", "on"])
@pytest.mark.parametrize("use_reentrant", [True, False], ids=["reentrant", "nonreentrant"])
class TestZero3TrainableRecomputeRetentionEvents(DistributedTest):
    world_size = 2

    def test_trainable_recompute_retention_events(self, use_reentrant, retain):
        device, _, _ = initialize_distributed()
        engine = _initialize(_CheckpointRetentionModel(8, use_reentrant), retain=retain)
        target = engine.module.first.weight
        assert target.requires_grad and not target.ds_persist and not target.is_external_param
        value = torch.randn(2, 8, device=device, requires_grad=True)

        with _record_target_lifecycle(engine, target) as events:
            engine.backward(engine(value).sum())
            _synchronize()

        recompute_releases = [event for event in events if event["kind"] == "recompute_forward_release"]
        backward_gathers = [event for event in events if event["kind"] == "gather" and not event["forward"]]
        matching_releases = [event for event in events if event["kind"] == "matching_backward_release"]
        assert recompute_releases, events

        if retain:
            assert all(event["before"]["status"] == ZeroParamStatus.AVAILABLE for event in recompute_releases)
            assert all(event["after"]["status"] == ZeroParamStatus.AVAILABLE for event in recompute_releases)
            for event in recompute_releases:
                module_id = event["before"]["owner_module"]
                synthetic_owner = -(module_id + 1)
                assert module_id in event["before"]["owners"], event
                assert synthetic_owner in event["before"]["owners"], event
                assert module_id in event["before"]["retained_modules"], event
                assert module_id not in event["after"]["owners"], event
                assert synthetic_owner in event["after"]["owners"], event
                assert module_id in event["after"]["retained_modules"], event
            assert not backward_gathers, events
            assert len(matching_releases) == 1, events
            matching_release = matching_releases[0]
            module_id = matching_release["before"]["owner_module"]
            synthetic_owner = -(module_id + 1)
            assert synthetic_owner in matching_release["before"]["owners"], matching_release
            assert module_id in matching_release["before"]["retained_modules"], matching_release
            assert synthetic_owner not in matching_release["after"]["owners"], matching_release
            assert module_id not in matching_release["after"]["retained_modules"], matching_release
            assert matching_release["after"]["status"] == ZeroParamStatus.NOT_AVAILABLE
        else:
            assert any(event["after"]["status"] == ZeroParamStatus.NOT_AVAILABLE for event in recompute_releases)
            assert backward_gathers, events
            assert not matching_releases, events

        assert value.grad is not None and torch.isfinite(value.grad).all()
        _assert_clean(engine)
        engine.destroy()


@pytest.mark.parametrize("control", ["trainable-persistent", "frozen", "trainable-external"])
class TestZero3TrainableRecomputeRetentionControls(DistributedTest):
    world_size = 2

    def test_retention_parameter_class_controls(self, control):
        device, _, _ = initialize_distributed()

        if control == "trainable-persistent":
            model = _CheckpointRetentionModel(8, use_reentrant=False)
            engine = _initialize(model, retain=True, persistence_threshold=1024)
            target = engine.module.first.weight
            assert target.requires_grad and target.ds_persist
        elif control == "frozen":
            model = _NoGradInputModel(8)
            engine = _initialize(model, retain=True)
            target = engine.module.frozen.weight
            assert not target.requires_grad and not target.ds_persist
        else:
            from unit.v1.zero.test_zero_activation_checkpoint_lifecycle import _ExternalCheckpointModel

            model = _ExternalCheckpointModel(8)
            model.producer.bias.requires_grad_(True)
            engine = _initialize(model, retain=True)
            target = engine.module.producer.bias
            assert target.requires_grad

        for _ in range(2):
            value = torch.randn(2, 8, device=device, requires_grad=(control != "frozen"))
            engine.backward(engine(value).sum())
            _synchronize()
            _assert_clean(engine, require_partitioned=(control != "trainable-external"))
            if control == "frozen":
                _assert_invocation_state_reset(engine)
            engine.step()

        if control == "trainable-persistent":
            assert target.ds_status == ZeroParamStatus.AVAILABLE
        elif control == "trainable-external":
            assert target.is_external_param
        else:
            assert target.ds_status == ZeroParamStatus.NOT_AVAILABLE
        engine.destroy()


@pytest.mark.parametrize("scenario", [
    "nested",
    "early-stop",
    "recompute-exception",
    "incomplete-backward",
    "gradient-accumulation",
    "multi-output-leaf-reentrant",
    "multi-output-leaf-nonreentrant",
])
class TestZero3TrainableRecomputeRetentionCleanup(DistributedTest):
    world_size = 2

    def test_retention_cleanup_paths(self, scenario):
        device, _, _ = initialize_distributed()

        if scenario == "nested":
            engine = _initialize(_RecursiveCheckpointModel(8), retain=True)
            for _ in range(2):
                value = torch.randn(2, 8, device=device, requires_grad=True)
                engine.backward(engine(value).sum())
                _synchronize()
                _assert_clean(engine)
                engine.step()
        elif scenario == "early-stop":
            model = _EarlyStopCheckpointModel(8)
            engine = _initialize(model, retain=True)
            for _ in range(2):
                value = torch.randn(2, 8, device=device, requires_grad=True)
                engine.backward(engine(value).sum())
                _synchronize()
                _assert_clean(engine)
                engine.step()
            assert model.recompute_started == 2
            assert model.recompute_reached_tail == 0
        elif scenario == "recompute-exception":
            model = _RecomputeExceptionModel(8)
            engine = _initialize(model, retain=True)
            value = torch.randn(2, 8, device=device, requires_grad=True)
            with pytest.raises(RuntimeError, match="injected checkpoint recompute forward"):
                engine.backward(engine(value).sum())
            parameter_offload = engine.optimizer.parameter_offload
            touched_modules = tuple(parameter_offload._recompute_grads_remaining_modules)
            assert touched_modules
            parameter_offload.release_backward_leftovers()
            parameter_offload.release_backward_leftovers()
            _assert_clean(engine)
            _assert_invocation_state_reset(engine, touched_modules)
            model.raise_during_recompute = False
            retry = torch.randn(2, 8, device=device, requires_grad=True)
            engine(retry).sum().backward()
            _synchronize()
            _assert_clean(engine)
            _assert_invocation_state_reset(engine, touched_modules)
        elif scenario == "incomplete-backward":
            model = _IncompleteBackwardModel(8)
            engine = _initialize(model, retain=True)
            value = torch.randn(2, 8, device=device, requires_grad=True)
            with pytest.raises(RuntimeError, match="injected incomplete checkpoint backward"):
                engine.backward(engine(value).sum())
            parameter_offload = engine.optimizer.parameter_offload
            touched_modules = tuple(parameter_offload._recompute_grads_remaining_modules)

            observed_reset = {"value": False}

            def observe_reset(unused_module, unused_inputs):
                _assert_clean(engine)
                _assert_invocation_state_reset(engine, touched_modules)
                observed_reset["value"] = True

            handle = engine.module.register_forward_pre_hook(observe_reset)
            retry = torch.randn(2, 8, device=device, requires_grad=True)
            try:
                engine(retry).sum().backward()
                _synchronize()
            finally:
                handle.remove()
            assert observed_reset["value"]
            _assert_clean(engine)
            _assert_invocation_state_reset(engine, touched_modules)
        elif scenario == "gradient-accumulation":
            engine = _initialize(_NoGradInputModel(8), retain=True, gradient_accumulation_steps=2)
            for _ in range(2):
                value = torch.randn(2, 8, device=device, requires_grad=False)
                engine(value).sum().backward()
                _synchronize()
                _assert_clean(engine)
                engine.step()
        else:
            use_reentrant = scenario.endswith("-reentrant") and not scenario.endswith("-nonreentrant")
            model = MultiTensorLeafFrozenModel(8, use_reentrant=use_reentrant)
            engine = _initialize(model, retain=True, leaf_types=[MultiTensorLeafBlock])
            value = torch.randn(2, 8, device=device, requires_grad=True)
            engine.backward(engine(value).sum())
            _synchronize()
            _assert_clean(engine)

        engine.destroy()


@pytest.mark.parametrize("use_reentrant", [True, False], ids=["reentrant", "nonreentrant"])
class TestZero3TrainableRecomputeRetentionModuleReuse(DistributedTest):
    world_size = 2

    def test_same_module_reuse_repeated_steps(self, use_reentrant):
        device, _, _ = initialize_distributed()
        torch.manual_seed(42)
        off = _initialize(_RepeatedModuleCheckpointRetentionModel(8, use_reentrant), retain=False)
        torch.manual_seed(42)
        on = _initialize(_RepeatedModuleCheckpointRetentionModel(8, use_reentrant), retain=True)
        off_target = off.module.shared.weight
        on_target = on.module.shared.weight

        for step in range(2):
            torch.manual_seed(123 + step)
            value = torch.randn(2, 8, device=device, requires_grad=True)
            with (_record_target_lifecycle(off, off_target) as off_events, _record_target_lifecycle(on, on_target) as
                  on_events):
                off_loss = off(value.clone()).sum()
                on_loss = on(value.clone()).sum()
                assert torch.isfinite(off_loss) and torch.isfinite(on_loss)
                torch.testing.assert_close(off_loss, on_loss)
                off.backward(off_loss)
                on.backward(on_loss)
                _synchronize()

            off_backward_gathers = [
                event for event in off_events if event["kind"] == "gather" and not event["forward"]
            ]
            on_backward_gathers = [event for event in on_events if event["kind"] == "gather" and not event["forward"]]
            assert off_backward_gathers, off_events
            assert not on_backward_gathers, on_events

            off_gradients = _collect_gradients(off)
            on_gradients = _collect_gradients(on)
            assert off_gradients.keys() == on_gradients.keys()
            assert "shared.weight" in on_gradients
            assert torch.count_nonzero(on_gradients["shared.weight"])
            for name in off_gradients:
                torch.testing.assert_close(off_gradients[name], on_gradients[name])

            _assert_clean(off)
            _assert_clean(on)
            _assert_invocation_state_reset(on, (on.module.shared, ))
            off.step()
            on.step()

        off.destroy()
        on.destroy()


@pytest.mark.parametrize("use_reentrant", [True, False], ids=["reentrant", "nonreentrant"])
class TestZero3TrainableRecomputeRetentionNested(DistributedTest):
    world_size = 2

    def test_nested_checkpoint_repeated_steps(self, use_reentrant):
        device, _, _ = initialize_distributed()
        engine = _initialize(_NestedCheckpointRetentionModel(8, use_reentrant), retain=True)

        for _ in range(2):
            value = torch.randn(2, 8, device=device, requires_grad=True)
            engine.backward(engine(value).sum())
            _synchronize()
            assert value.grad is not None and torch.isfinite(value.grad).all()
            _assert_clean(engine)
            engine.step()

        engine.destroy()


@pytest.mark.parametrize("use_reentrant", [True, False], ids=["reentrant", "nonreentrant"])
class TestZero3TrainableRecomputeRetentionParity(DistributedTest):
    world_size = 2

    def test_retention_numerical_parity(self, use_reentrant):
        device, _, _ = initialize_distributed()
        torch.manual_seed(42)
        off = _initialize(_CheckpointRetentionModel(8, use_reentrant), retain=False)
        torch.manual_seed(42)
        on = _initialize(_CheckpointRetentionModel(8, use_reentrant), retain=True)

        for step in range(2):
            torch.manual_seed(123 + step)
            value = torch.randn(2, 8, device=device, requires_grad=True)
            off_loss = off(value.clone()).sum()
            on_loss = on(value.clone()).sum()
            assert torch.isfinite(off_loss) and torch.isfinite(on_loss)
            torch.testing.assert_close(off_loss, on_loss)

            off.backward(off_loss)
            on.backward(on_loss)
            _synchronize()
            off_gradients = _collect_gradients(off)
            on_gradients = _collect_gradients(on)
            assert off_gradients.keys() == on_gradients.keys()
            for name in off_gradients:
                torch.testing.assert_close(off_gradients[name], on_gradients[name])
            _assert_clean(off)
            _assert_clean(on)
            off.step()
            on.step()

        off.destroy()
        on.destroy()

    def test_retention_is_noop_without_checkpoint_recompute(self, use_reentrant):
        device, _, _ = initialize_distributed()
        engine = _initialize(_PlainRetentionModel(8), retain=True)
        value = torch.randn(2, 8, device=device, requires_grad=True)

        engine.backward(engine(value).sum())
        _synchronize()

        assert value.grad is not None and torch.isfinite(value.grad).all()
        _assert_clean(engine)
        engine.destroy()
