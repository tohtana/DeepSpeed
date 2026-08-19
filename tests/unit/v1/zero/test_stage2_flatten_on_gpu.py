# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Test that ZeRO Stage 1 and 2 use the GPU flatten path when VRAM is sufficient.
Parametrized over zero_stage (1, 2) and dtype (fp32, fp16, bf16).
"""

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import set_log_level_from_string
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader

_DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class _MisalignedParamModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.ones(1))
        self.weight = torch.nn.Parameter(torch.ones(8, 8))

    def forward(self, x):
        return (x @ self.weight).sum() + self.offset.sum()


def _apply_dtype_to_config(config_dict, dtype):
    """Set bf16/fp16 in config_dict based on dtype; skip if not supported."""
    if dtype == "bf16":
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported on this accelerator")
        config_dict["bf16"] = {"enabled": True}
    elif dtype == "fp16":
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported on this accelerator")
        config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
    # fp32: no half-precision block


@pytest.mark.parametrize("zero_stage", [1, 2])
class TestStage12ParamAlignment(DistributedTest):
    world_size = 2

    def test_model_params_remain_16_byte_aligned(self, zero_stage):
        if not get_accelerator().is_available():
            pytest.skip("Accelerator not available")
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported on this accelerator")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage
            },
        }
        model = _MisalignedParamModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                               model=model,
                                               optimizer=optimizer,
                                               model_parameters=model.parameters())

        opt = engine.optimizer
        flat_views = opt.unflatten(opt.bit16_groups_flat[0], opt.round_robin_bit16_meta[0])
        assert flat_views[1].data_ptr() % 16 != 0
        assert engine.module.weight.data_ptr() % 16 == 0
        weight_before_step = engine.module.weight.detach().clone()

        data = torch.ones(1, 8, device=engine.device, dtype=torch.bfloat16)
        loss = engine(data)
        engine.backward(loss)
        engine.step()

        assert engine.module.weight.data_ptr() % 16 == 0
        assert not torch.equal(engine.module.weight, weight_before_step)
        flat_views = opt.unflatten(opt.bit16_groups_flat[0], opt.round_robin_bit16_meta[0])
        assert torch.equal(engine.module.weight, flat_views[1])

        # Universal checkpoint loading updates the fp32 partitions and then calls update_lp_params().
        # Verify that path also keeps standalone aligned model parameters synchronized with the flat buffer.
        for fp32_partition in opt.single_partition_of_fp32_groups:
            fp32_partition.data.add_(1)
        opt.update_lp_params()

        assert engine.module.weight.data_ptr() % 16 == 0
        flat_views = opt.unflatten(opt.bit16_groups_flat[0], opt.round_robin_bit16_meta[0])
        assert torch.equal(engine.module.weight, flat_views[1])


@pytest.mark.parametrize("zero_stage", [1, 2])
@pytest.mark.parametrize("load_kwargs", [{
    "load_module_only": True
}, {
    "load_optimizer_states": False
}],
                         ids=["module_only", "no_optimizer_states"])
class TestStage12ParamAlignmentCheckpointLoad(DistributedTest):
    world_size = 2

    def _build_engine(self, zero_stage, fill):
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage
            },
        }
        model = _MisalignedParamModel()
        with torch.no_grad():
            model.offset.fill_(fill)
            model.weight.fill_(fill)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                               model=model,
                                               optimizer=optimizer,
                                               model_parameters=model.parameters())
        return engine

    def _train_one_step(self, engine):
        data = torch.ones(1, 8, device=engine.device, dtype=torch.bfloat16)
        loss = engine(data)
        engine.backward(loss)
        engine.step()

    def test_flat_buffer_synchronized_after_checkpoint_load(self, tmpdir, zero_stage, load_kwargs):
        if not get_accelerator().is_available():
            pytest.skip("Accelerator not available")
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported on this accelerator")

        save_engine = self._build_engine(zero_stage, fill=1.0)
        save_opt = save_engine.optimizer
        assert save_opt.unflatten(save_opt.bit16_groups_flat[0],
                                  save_opt.round_robin_bit16_meta[0])[1].data_ptr() % 16 != 0

        self._train_one_step(save_engine)
        saved_weight = save_engine.module.weight.detach().clone()
        save_engine.save_checkpoint(str(tmpdir), tag="ckpt")

        # A clearly different starting point so a discarded load is unambiguous.
        load_engine = self._build_engine(zero_stage, fill=2.0)
        load_engine.load_checkpoint(str(tmpdir), tag="ckpt", **load_kwargs)

        weight_after_load = load_engine.module.weight.detach().clone()
        assert torch.equal(weight_after_load, saved_weight)

        # Checkpoint load writes the model parameter in place, and the fp32 master weights are
        # rebuilt by reading the flat buffer, so a standalone aligned parameter must be pushed back.
        load_opt = load_engine.optimizer
        flat_views = load_opt.unflatten(load_opt.bit16_groups_flat[0], load_opt.round_robin_bit16_meta[0])
        assert torch.equal(load_engine.module.weight, flat_views[1])

        # The loaded weights must survive the first optimizer step.
        self._train_one_step(load_engine)
        drift = (load_engine.module.weight.float() - weight_after_load.float()).abs().max().item()
        assert drift < 0.5


@pytest.mark.parametrize("zero_stage", [1, 2])
@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"], ids=["fp32", "fp16", "bf16"])
class TestStage2FlattenOnGPU(DistributedTest):
    """ZeRO-1 and ZeRO-2 with small model should flatten on GPU (sufficient VRAM)."""

    world_size = 2  # Run on 2 GPUs when available

    def test_flatten_on_gpu_path_taken(self, monkeypatch, zero_stage, dtype):
        """Assert the GPU flatten path was used (not CPU flatten + move)."""
        if not get_accelerator().is_available():
            pytest.skip("Accelerator not available")
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": zero_stage
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
        }
        _apply_dtype_to_config(config_dict, dtype)

        set_log_level_from_string("info")
        log_messages = []

        def mock_logger_info(msg, *args, **kwargs):
            log_messages.append(msg if isinstance(msg, str) else str(msg))

        monkeypatch.setattr("deepspeed.utils.logger.info", mock_logger_info)

        hidden_dim = 64
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
        )

        # Small model + no CPU offload => accelerator path logs "Flattening param group ... (sufficient memory)"
        accel_path_logs = [m for m in log_messages if "Flattening param group" in m and "(sufficient memory)" in m]
        assert accel_path_logs, (
            f"Expected accelerator flatten path (log should contain 'Flattening param group' and '(sufficient memory)'). "
            f"Captured messages: {log_messages}")

    def test_flat_buffers_on_accelerator(self, zero_stage, dtype):
        """Regression: flat buffers must end up on the accelerator (not left on CPU)."""
        if not get_accelerator().is_available():
            pytest.skip("Accelerator not available")
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": zero_stage
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
        }
        _apply_dtype_to_config(config_dict, dtype)

        hidden_dim = 64
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        engine, _, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
        )
        opt = engine.optimizer
        assert hasattr(opt, "bit16_groups_flat"), "ZeRO-1/2 optimizer should have bit16_groups_flat"
        device_type = get_accelerator().device_name()
        for i, flat in enumerate(opt.bit16_groups_flat):
            assert flat.device.type == device_type, (f"Flat buffer {i} must be on {device_type}, got {flat.device}")

    @pytest.mark.world_size(1)
    def test_flatten_on_accelerator_training_step(self, zero_stage, dtype):
        """Regression: flat buffer must be detached so inplace ops during step don't crash."""
        if not get_accelerator().is_available():
            pytest.skip("Accelerator not available")
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": zero_stage
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
        }
        _apply_dtype_to_config(config_dict, dtype)

        hidden_dim = 64
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        engine, _, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
        )
        for flat in engine.optimizer.bit16_groups_flat:
            assert flat.grad_fn is None, ("Flat buffer must be detached from autograd graph"
                                          " to prevent inplace-modification errors during optimizer step")

        data_loader = random_dataloader(model=engine,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=engine.device,
                                        dtype=_DTYPE_MAP[dtype])
        for batch in data_loader:
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()
