# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import pytest
import torch

import deepspeed
from deepspeed.runtime.zero import unwrap_model_for_generation
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest, preferred_dtype
from unit.simple_model import SimpleModel


class RootParameterModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.root_weight = torch.nn.Parameter(torch.eye(hidden_dim))
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = x @ self.root_weight
        return self.cross_entropy_loss(self.linear(x), y)


config = {
    "train_batch_size": 2,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015
        }
    },
    "zero_optimization": {
        "stage": 3,
        "stage3_param_persistence_threshold": 1,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        }
    }
}

if get_accelerator().is_bf16_supported():
    config["bf16"] = {"enabled": True}
elif get_accelerator().is_fp16_supported():
    config["fp16"] = {"enabled": True, "loss_scale": 138.}


def test_py20_class_bound_call_guard():
    """The fallback must intercept the class-bound dispatch used by PyTorch 2.0."""

    import deepspeed.runtime.zero.parameter_offload as parameter_offload

    class LegacyCallModule(torch.nn.Module):
        __call__ = torch.nn.Module._call_impl

        def forward(self, value):
            return value + 1

    module = LegacyCallModule()
    unguarded_module = LegacyCallModule()
    guard_events = []

    def guard(original_call, called_module, *args, **kwargs):
        guard_events.append("enter")
        try:
            return original_call(called_module, *args, **kwargs)
        except Exception:
            guard_events.append("error")
            raise

    state = parameter_offload._install_module_call_guard(module, guard)
    assert type(module) is LegacyCallModule
    assert unguarded_module(torch.ones(1)).item() == 2
    assert not guard_events
    hook = module.register_forward_pre_hook(lambda sub_module, inputs:
                                            (_ for _ in ()).throw(RuntimeError("legacy later pre-hook failed")))
    try:
        with pytest.raises(RuntimeError, match="legacy later pre-hook failed"):
            module(torch.ones(1))
        assert guard_events == ["enter", "error"]
    finally:
        hook.remove()
        parameter_offload._remove_module_call_guard(module, guard, state)

    assert LegacyCallModule.__dict__["__call__"] is torch.nn.Module._call_impl


class TestUnwrapModel(DistributedTest):
    # gather across more than 1 gpu
    world_size = 2

    def test(self):

        def hooks_exist(engine):
            if engine.optimizer is not None and hasattr(engine.optimizer, "parameter_offload"):
                optimizer_offload = engine.optimizer.parameter_offload
            elif engine.optimizer is not None:
                optimizer_offload = engine.optimizer

            hooks = 0
            for hook in optimizer_offload.forward_hooks:
                hooks += 1
            if hooks > 0:
                return True
            return False

        model = SimpleModel(hidden_dim=100)
        engine, _, _, _ = deepspeed.initialize(args=None, model=model, config=config)

        with unwrap_model_for_generation(engine):
            # assert no hooks
            assert not hooks_exist(engine)
            # assert parameters gathered
            assert model.linears[0].weight.numel() != 0, "GatheredParameters should give a non-0-sized tensor"

        # assert hooks
        assert hooks_exist(engine)


class TestUnwrapModelTraceInvalidate(DistributedTest):
    # unwrap_model_for_generation re-registers the ZeRO-3 hooks; without trace
    # invalidation the next training step pops an empty fetch deque.
    world_size = 2

    def test(self):
        model = SimpleModel(hidden_dim=100)
        engine, _, _, _ = deepspeed.initialize(args=None, model=model, config=config)

        x = torch.randn(2, 100, device=engine.device, dtype=preferred_dtype())
        y = torch.empty(2, dtype=torch.long, device=engine.device).random_(100)

        loss = engine(x, y)
        engine.backward(loss)
        engine.step()

        with unwrap_model_for_generation(engine):
            pass

        loss = engine(x, y)
        engine.backward(loss)
        engine.step()


@pytest.mark.parametrize("force_py20_fallback", [False, True])
class TestUnwrapModelRootHookOrder(DistributedTest):
    world_size = 2

    def test(self, force_py20_fallback):
        import deepspeed.runtime.zero.parameter_offload as parameter_offload

        actual_always_call_support = parameter_offload.FORWARD_HOOK_ALWAYS_CALL_SUPPORTED
        if force_py20_fallback:
            parameter_offload.FORWARD_HOOK_ALWAYS_CALL_SUPPORTED = False

        engine = None
        original_reset_step = None
        original_pre_forward = None
        try:
            hidden_dim = 8
            model = RootParameterModel(hidden_dim)
            engine, _, _, _ = deepspeed.initialize(args=None, model=model, config=config)
            offload = engine.optimizer.parameter_offload
            coordinator = offload.get_param_coordinator()

            x = torch.randn(2, hidden_dim, device=engine.device, dtype=preferred_dtype())
            y = torch.empty(2, dtype=torch.long, device=engine.device).random_(hidden_dim)

            loss = engine(x, y)
            engine.backward(loss)
            engine.step()

            with unwrap_model_for_generation(engine):
                pass

            events = []
            original_reset_step = coordinator.reset_step
            original_pre_forward = offload.pre_sub_module_forward_function

            def observed_reset_step():
                events.append("reset")
                return original_reset_step()

            def observed_pre_forward(sub_module):
                if sub_module is engine.module:
                    events.append("root_fetch")
                return original_pre_forward(sub_module)

            coordinator.reset_step = observed_reset_step
            offload.pre_sub_module_forward_function = observed_pre_forward

            for _ in range(3):
                events.clear()
                loss = engine(x, y)
                assert events[:2] == ["reset", "root_fetch"]
                assert torch.isfinite(loss.detach())
                engine.backward(loss)
                engine.step()

                nonpersistent_params = [param for param in engine.module.parameters() if not param.ds_persist]
                resident_numel = sum(param.ds_numel for param in nonpersistent_params
                                     if param.ds_status != ZeroParamStatus.NOT_AVAILABLE)
                available_numel = coordinator._PartitionedParameterCoordinator__n_available_params
                assert available_numel == resident_numel
        finally:
            if engine is not None:
                if original_reset_step is not None:
                    engine.optimizer.parameter_offload.get_param_coordinator().reset_step = original_reset_step
                if original_pre_forward is not None:
                    engine.optimizer.parameter_offload.pre_sub_module_forward_function = original_pre_forward
                engine.destroy()
            parameter_offload.FORWARD_HOOK_ALWAYS_CALL_SUPPORTED = actual_always_call_support


@pytest.mark.parametrize("force_py20_fallback", [False, True])
class TestUnwrapModelExceptionRestore(DistributedTest):
    world_size = 2

    def test(self, force_py20_fallback):
        import deepspeed.runtime.zero.parameter_offload as parameter_offload

        actual_always_call_support = parameter_offload.FORWARD_HOOK_ALWAYS_CALL_SUPPORTED
        if force_py20_fallback:
            parameter_offload.FORWARD_HOOK_ALWAYS_CALL_SUPPORTED = False

        engine = None
        try:
            hidden_dim = 8
            model = RootParameterModel(hidden_dim)
            engine, _, _, _ = deepspeed.initialize(args=None, model=model, config=config)
            offload = engine.optimizer.parameter_offload

            x = torch.randn(2, hidden_dim, device=engine.device, dtype=preferred_dtype())
            y = torch.empty(2, dtype=torch.long, device=engine.device).random_(hidden_dim)

            loss = engine(x, y)
            engine.backward(loss)
            engine.step()

            wrapper_count = len(offload.forward_wrappers)
            with pytest.raises(RuntimeError, match="generation failed"):
                with unwrap_model_for_generation(engine):
                    assert not offload.forward_hooks
                    assert offload.fwd_pre_hook is None
                    assert not offload.forward_wrappers
                    raise RuntimeError("generation failed")

            assert offload.forward_hooks
            assert offload.fwd_pre_hook is not None
            if force_py20_fallback:
                assert len(offload.forward_wrappers) == wrapper_count

            loss = engine(x, y)
            assert torch.isfinite(loss.detach())
            engine.backward(loss)
            engine.step()
        finally:
            if engine is not None:
                engine.destroy()
            parameter_offload.FORWARD_HOOK_ALWAYS_CALL_SUPPORTED = actual_always_call_support


class TestZeroNativeEarlierPreHookFailure(DistributedTest):
    world_size = 2

    def test(self):
        import deepspeed.runtime.zero.parameter_offload as parameter_offload

        if not parameter_offload.FORWARD_HOOK_ALWAYS_CALL_SUPPORTED:
            pytest.skip("native always_call forward hooks are unavailable")

        engine = None
        global_pre_hook = None
        original_pre_forward = None
        try:
            hidden_dim = 8
            model = RootParameterModel(hidden_dim)
            engine, _, _, _ = deepspeed.initialize(args=None, model=model, config=config)
            offload = engine.optimizer.parameter_offload
            coordinator = offload.get_param_coordinator()

            x = torch.randn(2, hidden_dim, device=engine.device, dtype=preferred_dtype())
            y = torch.empty(2, dtype=torch.long, device=engine.device).random_(hidden_dim)

            # Establish a normal trace and partition the root-owned parameter before
            # injecting a global hook, which PyTorch runs before ZeRO's local hooks.
            loss = engine(x, y)
            engine.backward(loss)
            engine.step()

            zero_pre_calls = 0
            original_pre_forward = offload.pre_sub_module_forward_function

            def observed_pre_forward(sub_module):
                nonlocal zero_pre_calls
                zero_pre_calls += 1
                return original_pre_forward(sub_module)

            def raise_before_zero(sub_module, inputs):
                if sub_module is engine.module:
                    raise RuntimeError("earlier pre-hook failed")

            offload.pre_sub_module_forward_function = observed_pre_forward
            global_pre_hook = torch.nn.modules.module.register_module_forward_pre_hook(raise_before_zero)
            calls_before_failure = zero_pre_calls
            with pytest.raises(RuntimeError, match="earlier pre-hook failed"):
                engine(x, y)
            assert zero_pre_calls == calls_before_failure

            global_pre_hook.remove()
            global_pre_hook = None

            def assert_balanced():
                assert parameter_offload.FWD_MODULE_STACK == [engine.module]
                assert all(not sub_module._external_params for sub_module in engine.module.modules())
                nonpersistent_params = [param for param in engine.module.parameters() if not param.ds_persist]
                resident_numel = sum(param.ds_numel for param in nonpersistent_params
                                     if param.ds_status != ZeroParamStatus.NOT_AVAILABLE)
                available_numel = coordinator._PartitionedParameterCoordinator__n_available_params
                assert available_numel == resident_numel

            assert_balanced()
            for _ in range(2):
                loss = engine(x, y)
                assert torch.isfinite(loss.detach())
                engine.backward(loss)
                engine.step()
                assert_balanced()
        finally:
            if global_pre_hook is not None:
                global_pre_hook.remove()
            if engine is not None:
                if original_pre_forward is not None:
                    engine.optimizer.parameter_offload.pre_sub_module_forward_function = original_pre_forward
                engine.destroy()


class TestZeroNativeEarlierPostHookFailure(DistributedTest):
    world_size = 2

    def test(self):
        import deepspeed.runtime.zero.parameter_offload as parameter_offload

        if not parameter_offload.FORWARD_HOOK_ALWAYS_CALL_SUPPORTED:
            pytest.skip("native always_call forward hooks are unavailable")

        engine = None
        global_post_hook = None
        try:
            hidden_dim = 8
            model = RootParameterModel(hidden_dim)
            engine, _, _, _ = deepspeed.initialize(args=None, model=model, config=config)
            offload = engine.optimizer.parameter_offload
            coordinator = offload.get_param_coordinator()

            x = torch.randn(2, hidden_dim, device=engine.device, dtype=preferred_dtype())
            y = torch.empty(2, dtype=torch.long, device=engine.device).random_(hidden_dim)

            # Record a trace so the failing forward can prefetch parameters owned by
            # modules that have not run yet. Global forward hooks execute before the
            # local ZeRO always-call hook and receive the valid root output.
            loss = engine(x, y)
            engine.backward(loss)
            engine.step()

            root_output_was_tensor = False

            def raise_before_zero_post(sub_module, inputs, output):
                nonlocal root_output_was_tensor
                if sub_module is engine.module:
                    root_output_was_tensor = torch.is_tensor(output)
                    raise RuntimeError("earlier post-hook failed")

            global_post_hook = torch.nn.modules.module.register_module_forward_hook(raise_before_zero_post)
            with pytest.raises(RuntimeError, match="earlier post-hook failed"):
                engine(x, y)
            assert root_output_was_tensor

            global_post_hook.remove()
            global_post_hook = None

            def assert_balanced():
                assert parameter_offload.FWD_MODULE_STACK == [engine.module]
                assert all(not sub_module._external_params for sub_module in engine.module.modules())
                assert all(not param.ds_active_sub_modules for param in engine.module.parameters())
                nonpersistent_params = [param for param in engine.module.parameters() if not param.ds_persist]
                resident_numel = sum(param.ds_numel for param in nonpersistent_params
                                     if param.ds_status != ZeroParamStatus.NOT_AVAILABLE)
                available_numel = coordinator._PartitionedParameterCoordinator__n_available_params
                assert available_numel == resident_numel

            assert_balanced()
            for _ in range(2):
                loss = engine(x, y)
                assert torch.isfinite(loss.detach())
                engine.backward(loss)
                engine.step()
                assert_balanced()
        finally:
            if global_post_hook is not None:
                global_post_hook.remove()
            if engine is not None:
                engine.destroy()


@pytest.mark.parametrize("force_py20_fallback", [False, True])
class TestZeroLaterPreHookFailure(DistributedTest):
    world_size = 2

    def test(self, force_py20_fallback):
        import deepspeed.runtime.zero.parameter_offload as parameter_offload

        actual_always_call_support = parameter_offload.FORWARD_HOOK_ALWAYS_CALL_SUPPORTED
        if not force_py20_fallback and not actual_always_call_support:
            pytest.skip("native always_call forward hooks are unavailable")
        if force_py20_fallback:
            parameter_offload.FORWARD_HOOK_ALWAYS_CALL_SUPPORTED = False
        engine = None
        later_pre_hook = None
        original_pre_forward = None
        try:
            hidden_dim = 8
            model = RootParameterModel(hidden_dim)
            engine, _, _, _ = deepspeed.initialize(args=None, model=model, config=config)
            offload = engine.optimizer.parameter_offload
            coordinator = offload.get_param_coordinator()

            x = torch.randn(2, hidden_dim, device=engine.device, dtype=preferred_dtype())
            y = torch.empty(2, dtype=torch.long, device=engine.device).random_(hidden_dim)

            loss = engine(x, y)
            engine.backward(loss)
            engine.step()

            zero_pre_calls = 0
            original_pre_forward = offload.pre_sub_module_forward_function

            def observed_pre_forward(sub_module):
                nonlocal zero_pre_calls
                zero_pre_calls += 1
                return original_pre_forward(sub_module)

            def raise_after_zero(sub_module, inputs):
                if sub_module is engine.module:
                    raise RuntimeError("later pre-hook failed")

            offload.pre_sub_module_forward_function = observed_pre_forward
            later_pre_hook = engine.module.register_forward_pre_hook(raise_after_zero)
            calls_before_failure = zero_pre_calls
            with pytest.raises(RuntimeError, match="later pre-hook failed"):
                engine(x, y)
            assert zero_pre_calls == calls_before_failure + 1

            later_pre_hook.remove()
            later_pre_hook = None

            def assert_balanced():
                assert parameter_offload.FWD_MODULE_STACK == [engine.module]
                assert all(not sub_module._external_params for sub_module in engine.module.modules())
                assert all(not param.ds_active_sub_modules for param in engine.module.parameters())
                nonpersistent_params = [param for param in engine.module.parameters() if not param.ds_persist]
                resident_numel = sum(param.ds_numel for param in nonpersistent_params
                                     if param.ds_status != ZeroParamStatus.NOT_AVAILABLE)
                available_numel = coordinator._PartitionedParameterCoordinator__n_available_params
                assert available_numel == resident_numel

            assert_balanced()
            for _ in range(2):
                loss = engine(x, y)
                assert torch.isfinite(loss.detach())
                engine.backward(loss)
                engine.step()
                assert_balanced()
        finally:
            if later_pre_hook is not None:
                later_pre_hook.remove()
            if engine is not None:
                if original_pre_forward is not None:
                    engine.optimizer.parameter_offload.pre_sub_module_forward_function = original_pre_forward
                engine.destroy()
            parameter_offload.FORWARD_HOOK_ALWAYS_CALL_SUPPORTED = actual_always_call_support


@pytest.mark.parametrize("with_optimizer", [True, False])
class TestHybridEngineFallbackDestroy(DistributedTest):
    world_size = 2

    def test(self, with_optimizer):
        import deepspeed.runtime.zero.parameter_offload as parameter_offload
        from transformers import OPTConfig, OPTModel

        actual_always_call_support = parameter_offload.FORWARD_HOOK_ALWAYS_CALL_SUPPORTED
        parameter_offload.FORWARD_HOOK_ALWAYS_CALL_SUPPORTED = False
        engine = None
        destroyed = False
        try:
            model_config = OPTConfig(vocab_size=32,
                                     hidden_size=8,
                                     ffn_dim=16,
                                     num_hidden_layers=1,
                                     num_attention_heads=2,
                                     max_position_embeddings=8,
                                     word_embed_proj_dim=8)
            model = OPTModel(model_config)
            model.generate = model.forward
            hybrid_config = copy.deepcopy(config)
            hybrid_config["zero_optimization"]["stage3_param_persistence_threshold"] = 1_000_000
            hybrid_config["hybrid_engine"] = {"enabled": True, "pin_parameters": True}
            if not with_optimizer:
                del hybrid_config["optimizer"]
            engine, _, _, _ = deepspeed.initialize(args=None, model=model, config=hybrid_config)
            if with_optimizer:
                assert hasattr(engine.optimizer, "parameter_offload")
            else:
                assert not hasattr(engine.optimizer, "parameter_offload")
            offload = getattr(engine.optimizer, "parameter_offload", engine.optimizer)

            assert offload.forward_wrappers
            assert engine._inference_containers
            orig_module = engine._orig_modules[0]
            outer_wrapper = orig_module.forward
            assert offload._get_forward_delegate(orig_module) is engine._orig_fwds[0]

            engine.eval()
            inference_delegate = offload._get_forward_delegate(orig_module)
            assert orig_module.forward is outer_wrapper
            assert inference_delegate is not engine._orig_fwds[0]

            engine.destroy()
            destroyed = True
            assert not offload.forward_wrappers
            assert not offload.forward_call_wrappers
            assert not offload.forward_wrapper_states
            assert orig_module.forward is inference_delegate
        finally:
            if engine is not None and not destroyed:
                engine.train()
                engine.destroy()
            parameter_offload.FORWARD_HOOK_ALWAYS_CALL_SUPPORTED = actual_always_call_support
