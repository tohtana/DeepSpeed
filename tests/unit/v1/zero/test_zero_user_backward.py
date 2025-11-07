# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from copy import deepcopy

from unit.common import DistributedTest, preferred_dtype, allclose_on_all_ranks
from unit.simple_model import SimpleModel, random_dataloader
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import safe_get_full_grad


class SimpleNonScalarModel(torch.nn.Module):
    """Model that returns non-scalar output for testing tensor.backward(grad)"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # Returns non-scalar output
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def get_config_dict(zero_stage, allow_user_backward=False, gradient_accumulation_steps=1):
    """Helper to create config dict with common settings"""
    config_dict = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage,
            "allow_user_backward": allow_user_backward,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
    }

    if zero_stage == 3:
        # For ZeRO-3, force partitioning of all parameters
        config_dict["zero_optimization"]["stage3_param_persistence_threshold"] = 0

    if get_accelerator().is_bf16_supported():
        config_dict["bf16"] = {"enabled": True}
    elif get_accelerator().is_fp16_supported():
        config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}

    return config_dict


def collect_gradients_safe(model):
    """Collect gradients from model parameters using safe_get_full_grad API"""
    grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad = safe_get_full_grad(param)
            if grad is not None:
                grads[name] = grad.detach().clone().cpu()
    return grads


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestZeroUserBackwardBasic(DistributedTest):
    """Test basic functionality of allow_user_backward feature"""
    world_size = 2

    def test_loss_backward_matches_engine_backward(self, zero_stage):
        """Test that loss.backward() produces same gradients as engine.backward(loss)"""
        hidden_dim = 4

        # Create two identical models
        model1 = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        model2 = deepcopy(model1)

        # Initialize with engine.backward (traditional way)
        config1 = get_config_dict(zero_stage, allow_user_backward=False)
        model_engine1, _, _, _ = deepspeed.initialize(
            config=config1,
            model=model1,
            model_parameters=model1.parameters()
        )

        # Initialize with loss.backward (new way)
        config2 = get_config_dict(zero_stage, allow_user_backward=True)
        model_engine2, _, _, _ = deepspeed.initialize(
            config=config2,
            model=model2,
            model_parameters=model2.parameters()
        )

        data_loader = random_dataloader(
            model=model_engine1,
            total_samples=8,
            hidden_dim=hidden_dim,
            device=model_engine1.device
        )

        # Run one training step with engine.backward
        batch = next(iter(data_loader))
        loss1 = model_engine1(batch[0], batch[1])
        model_engine1.backward(loss1)
        grads1 = collect_gradients_safe(model_engine1)

        # Run one training step with loss.backward
        loss2 = model_engine2(batch[0], batch[1])
        loss2.backward()
        grads2 = collect_gradients_safe(model_engine2)

        # Compare gradients across all ranks
        assert len(grads1) == len(grads2), "Different number of parameters with gradients"
        for name in grads1.keys():
            assert name in grads2, f"Parameter {name} missing in grads2"
            allclose_on_all_ranks(grads1[name], grads2[name],
                                  rtol=1e-4, atol=1e-5,
                                  assert_message=f"Gradients differ for parameter {name}")

        model_engine1.destroy()
        model_engine2.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestZeroUserBackwardNonScalar(DistributedTest):
    """Test non-scalar backward support"""
    world_size = 2

    def test_non_scalar_backward(self, zero_stage):
        """Test that tensor.backward(grad) works correctly by comparing with PyTorch baseline"""
        hidden_dim = 4
        batch_size = 2
        lr = 1e-3

        # Create two identical models - one for PyTorch baseline, one for DeepSpeed
        torch.manual_seed(42)
        model_pytorch = SimpleNonScalarModel(hidden_dim=hidden_dim)
        model_pytorch = model_pytorch.to(get_accelerator().device_name())

        torch.manual_seed(42)
        model_deepspeed = SimpleNonScalarModel(hidden_dim=hidden_dim)

        # Initialize DeepSpeed with allow_user_backward
        config = get_config_dict(zero_stage, allow_user_backward=True)
        model_engine, _, _, _ = deepspeed.initialize(
            config=config,
            model=model_deepspeed,
            model_parameters=model_deepspeed.parameters()
        )

        # Convert PyTorch model to the same dtype as DeepSpeed
        dtype = preferred_dtype()
        model_pytorch = model_pytorch.to(dtype)

        # Create PyTorch optimizer with same settings
        optimizer_pytorch = torch.optim.Adam(model_pytorch.parameters(), lr=lr)

        # Create same input for both models
        torch.manual_seed(123)
        x = torch.randn(batch_size, hidden_dim, device=get_accelerator().device_name(), dtype=dtype)

        # PyTorch baseline: forward, backward with gradient
        output_pytorch = model_pytorch(x)
        grad_output = torch.ones_like(output_pytorch)
        output_pytorch.backward(grad_output)

        # Collect PyTorch gradients after backward, before step
        pytorch_grads = {}
        for name, param in model_pytorch.named_parameters():
            if param.grad is not None:
                pytorch_grads[name] = param.grad.detach().clone().cpu()

        # DeepSpeed: forward, backward with gradient
        output_deepspeed = model_engine(x)
        grad_output_ds = torch.ones_like(output_deepspeed)
        output_deepspeed.backward(grad_output_ds)

        # Collect DeepSpeed gradients after backward, before step using safe API
        deepspeed_grads = {}
        for name, param in model_engine.named_parameters():
            # Remove 'module.' prefix if present (DeepSpeed wraps the model)
            clean_name = name.replace('module.', '')
            grad = safe_get_full_grad(param)
            if grad is not None:
                deepspeed_grads[clean_name] = grad.detach().clone().cpu()

        # Compare gradients across all ranks
        assert len(pytorch_grads) == len(deepspeed_grads), \
            f"Gradient count mismatch: PyTorch has {len(pytorch_grads)}, DeepSpeed has {len(deepspeed_grads)}"

        for name in pytorch_grads.keys():
            assert name in deepspeed_grads, f"Gradient for parameter {name} missing in DeepSpeed model"

            # Gradients should match between PyTorch and DeepSpeed
            # Note: DeepSpeed may accumulate gradients in fp32 even when model is bf16,
            # so we convert both to fp32 for comparison
            pytorch_grad_fp32 = pytorch_grads[name].float()
            deepspeed_grad_fp32 = deepspeed_grads[name].float()
            allclose_on_all_ranks(pytorch_grad_fp32, deepspeed_grad_fp32,
                                  rtol=1e-3, atol=1e-4,
                                  assert_message=f"Gradient for parameter {name} mismatch between PyTorch and DeepSpeed after non-scalar backward")

        # Now run optimizer step
        optimizer_pytorch.step()
        model_engine.step()

        # Collect PyTorch parameters after step
        pytorch_params = {}
        for name, param in model_pytorch.named_parameters():
            pytorch_params[name] = param.detach().clone().cpu()

        # Collect DeepSpeed parameters after step
        deepspeed_params = {}
        for name, param in model_engine.named_parameters():
            # Remove 'module.' prefix if present (DeepSpeed wraps the model)
            clean_name = name.replace('module.', '')
            if zero_stage == 3:
                with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                    deepspeed_params[clean_name] = param.detach().clone().cpu()
            else:
                deepspeed_params[clean_name] = param.detach().clone().cpu()

        # Compare parameters across all ranks
        assert len(pytorch_params) == len(deepspeed_params), \
            f"Parameter count mismatch: PyTorch has {len(pytorch_params)}, DeepSpeed has {len(deepspeed_params)}"

        for name in pytorch_params.keys():
            assert name in deepspeed_params, f"Parameter {name} missing in DeepSpeed model"

            # Parameters should match between PyTorch and DeepSpeed
            # Convert to fp32 for comparison in case of dtype mismatch
            pytorch_param_fp32 = pytorch_params[name].float()
            deepspeed_param_fp32 = deepspeed_params[name].float()
            allclose_on_all_ranks(pytorch_param_fp32, deepspeed_param_fp32,
                                  rtol=1e-3, atol=1e-4,
                                  assert_message=f"Parameter {name} mismatch between PyTorch and DeepSpeed after non-scalar backward")

        model_engine.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestZeroUserBackwardDisabled(DistributedTest):
    """Test error handling when allow_user_backward is disabled"""
    world_size = 1

    def test_error_when_disabled(self, zero_stage):
        """Test that proper error is raised when calling loss.backward() without enabling the feature"""
        hidden_dim = 4

        model = SimpleModel(hidden_dim=hidden_dim)

        # Initialize without allow_user_backward
        config = get_config_dict(zero_stage, allow_user_backward=False)
        model_engine, _, _, _ = deepspeed.initialize(
            config=config,
            model=model,
            model_parameters=model.parameters()
        )

        data_loader = random_dataloader(
            model=model_engine,
            total_samples=4,
            hidden_dim=hidden_dim,
            device=model_engine.device
        )

        batch = next(iter(data_loader))
        loss = model_engine(batch[0], batch[1])

        # Calling loss.backward() should raise an error
        with pytest.raises(RuntimeError, match="DeepSpeed requires backward to be invoked via"):
            loss.backward()

        model_engine.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestZeroUserBackwardGradAccumulation(DistributedTest):
    """Test gradient accumulation with user backward"""
    world_size = 2

    def test_grad_accumulation(self, zero_stage):
        """Test that gradient accumulation works correctly with loss.backward()"""
        hidden_dim = 4
        gradient_accumulation_steps = 4

        _run_baseline = True
        _run_new = True

        # Create two identical models
        model1 = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        model2 = deepcopy(model1)

        # Initialize with engine.backward (traditional way)
        config1 = get_config_dict(zero_stage, allow_user_backward=False,
                                    gradient_accumulation_steps=gradient_accumulation_steps)
        model_engine1, _, _, _ = deepspeed.initialize(
            config=config1,
            model=model1,
            model_parameters=model1.parameters()
        )

        # Initialize with loss.backward (new way)
        config2 = get_config_dict(zero_stage, allow_user_backward=True,
                                gradient_accumulation_steps=gradient_accumulation_steps)

        model_engine2, _, _, _ = deepspeed.initialize(
            config=config2,
            model=model2,
            model_parameters=model2.parameters()
        )

        data_loader = random_dataloader(
            model=model_engine2 if _run_new else model_engine1,
            total_samples=16,
            hidden_dim=hidden_dim,
            device=model_engine2.device if _run_new else model_engine1.device
        )

        # Run training with gradient accumulation
        for i, batch in enumerate(data_loader):
            # Traditional way
            loss1 = model_engine1(batch[0], batch[1])
            model_engine1.backward(loss1)

            # New way
            loss2 = model_engine2(batch[0], batch[1])
            loss2.backward()

            # Compare gradients before optimizer step
            # Using safe_get_full_grad API handles all ZeRO stages correctly
            if model_engine1.is_gradient_accumulation_boundary():
                grads1 = collect_gradients_safe(model_engine1)
                grads2 = collect_gradients_safe(model_engine2)

                # Compare gradients
                assert len(grads1) == len(grads2), f"Different number of parameters with gradients at step {i}"
                for name in grads1.keys():
                    assert name in grads2, f"Parameter {name} missing in grads2 at step {i}"
                    allclose_on_all_ranks(grads2[name], grads1[name],
                                        rtol=1e-4, atol=1e-5,
                                        assert_message=f"Gradients differ for {name} at step {i}")

            # Now step both models
            model_engine1.step()
            model_engine2.step()

        model_engine1.destroy()
        model_engine2.destroy()


@pytest.mark.parametrize("zero_stage", [1])
class TestZeroUserBackwardMultipleBackward(DistributedTest):
    """Test multiple backward calls in one forward pass"""
    world_size = 2

    def test_multiple_backward_calls(self, zero_stage):
        """Test calling backward multiple times with retain_graph=True"""
        hidden_dim = 4

        model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)

        config = get_config_dict(zero_stage, allow_user_backward=True)
        model_engine, _, _, _ = deepspeed.initialize(
            config=config,
            model=model,
            model_parameters=model.parameters()
        )

        data_loader = random_dataloader(
            model=model_engine,
            total_samples=4,
            hidden_dim=hidden_dim,
            device=model_engine.device
        )

        batch = next(iter(data_loader))
        loss = model_engine(batch[0], batch[1])

        # First backward with retain_graph=True
        loss.backward(retain_graph=True)
        grads_first = collect_gradients_safe(model_engine)

        # Second backward should accumulate gradients
        loss.backward()
        grads_second = collect_gradients_safe(model_engine)

        # Gradients should be accumulated (doubled)
        for name in grads_first.keys():
            expected = grads_first[name] * 2
            allclose_on_all_ranks(grads_second[name], expected,
                                  rtol=1e-4, atol=1e-5,
                                  assert_message=f"Gradient accumulation failed for parameter {name}")

        model_engine.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestZeroUserBackwardCompatibility(DistributedTest):
    """Test that engine.backward still works when allow_user_backward is enabled"""
    world_size = 2

    def test_engine_backward_still_works(self, zero_stage):
        """Test that engine.backward() still works correctly when allow_user_backward=True"""
        hidden_dim = 4

        model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)

        config = get_config_dict(zero_stage, allow_user_backward=True)
        model_engine, _, _, _ = deepspeed.initialize(
            config=config,
            model=model,
            model_parameters=model.parameters()
        )

        data_loader = random_dataloader(
            model=model_engine,
            total_samples=8,
            hidden_dim=hidden_dim,
            device=model_engine.device
        )

        # Run several training steps using engine.backward
        for i, batch in enumerate(data_loader):
            loss = model_engine(batch[0], batch[1])
            # Should still work with engine.backward
            model_engine.backward(loss)

            # Verify gradients are computed
            if model_engine.is_gradient_accumulation_boundary():
                grads = collect_gradients_safe(model_engine)
                assert len(grads) > 0, f"No gradients computed at step {i}"

            model_engine.step()

        model_engine.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestZeroUserBackwardMixedLoss(DistributedTest):
    """Test using separate loss functions"""
    world_size = 2

    def test_separate_loss_function(self, zero_stage):
        """Test using loss function defined separately from model"""
        hidden_dim = 4

        class SimpleOutputModel(torch.nn.Module):
            """Model that returns output without computing loss"""
            def __init__(self, hidden_dim):
                super().__init__()
                self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
                self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        model = SimpleOutputModel(hidden_dim=hidden_dim)

        config = get_config_dict(zero_stage, allow_user_backward=True)
        model_engine, _, _, _ = deepspeed.initialize(
            config=config,
            model=model,
            model_parameters=model.parameters()
        )

        # Define loss function separately
        loss_fn = torch.nn.CrossEntropyLoss()

        # Create data
        batch_size = 2
        x = torch.randn(batch_size, hidden_dim, device=model_engine.device, dtype=preferred_dtype())
        y = torch.randint(0, hidden_dim, (batch_size,), device=model_engine.device)

        # Forward pass
        output = model_engine(x)

        # Compute loss outside model
        loss = loss_fn(output, y)

        # Backward should work
        loss.backward()

        # Check gradients
        grads = collect_gradients_safe(model_engine)
        assert len(grads) > 0, "No gradients computed"

        model_engine.destroy()
