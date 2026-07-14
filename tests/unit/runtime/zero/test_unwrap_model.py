# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

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


class TestUnwrapModelRootHookOrder(DistributedTest):
    world_size = 2

    def test(self):
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
