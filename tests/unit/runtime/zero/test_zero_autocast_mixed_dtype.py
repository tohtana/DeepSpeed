# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from contextlib import contextmanager

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.torch_autocast import get_comm_dtype, has_comm_dtype
from deepspeed.runtime.zero.partition_parameters import get_allgather_dtype
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad
from unit.common import DistributedTest
from unit.util import bf16_required_version_check


def _safe_module_name():
    return f"{MixedDtypeAdapterModule.__module__}.{MixedDtypeAdapterModule.__name__}"


def _zero3_bf16_autocast_config():
    return {
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "bf16": {
            "enabled": True,
            "bf16_master_weights_and_grads": True,
            "bf16_optimizer_states": True,
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_param_persistence_threshold": 0,
            "stage3_module_granularity_threshold": 0,
            "stage3_use_all_reduce_for_fetch_params": False,
        },
        "torch_autocast": {
            "enabled": True,
            "dtype": str(torch.bfloat16),
            "lower_precision_safe_modules": [_safe_module_name()],
        },
    }


class MixedDtypeAdapterModule(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_weight = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01, requires_grad=False)

    def attach_fp32_adapter(self, rank):
        device = get_accelerator().current_device_name()
        self.adapter_a = torch.nn.Parameter(
            torch.randn(rank, self.hidden_dim, device=device, dtype=torch.float32) * 0.01)
        self.adapter_b = torch.nn.Parameter(
            torch.randn(self.hidden_dim, rank, device=device, dtype=torch.float32) * 0.01)

        assert hasattr(self.base_weight, "convert_to_zero_parameters")
        self.base_weight.convert_to_zero_parameters([self.adapter_a, self.adapter_b])

    def forward(self, x, target):
        base = torch.nn.functional.linear(x, self.base_weight)
        adapter_hidden = torch.nn.functional.linear(x, self.adapter_a)
        adapter = torch.nn.functional.linear(adapter_hidden, self.adapter_b) / self.adapter_a.shape[0]
        output = base + adapter
        return torch.nn.functional.mse_loss(output.float(), target.float())


def _assert_mixed_partition_dtypes(model):
    assert model.base_weight.dtype == torch.bfloat16
    assert model.base_weight.ds_tensor.dtype == torch.bfloat16

    for adapter_param in [model.adapter_a, model.adapter_b]:
        assert adapter_param.dtype == torch.float32
        assert adapter_param.ds_tensor.dtype == torch.float32


def _assert_autocast_comm_dtype(model):
    for param in [model.base_weight, model.adapter_a, model.adapter_b]:
        assert has_comm_dtype(param)
        assert get_comm_dtype(param) == torch.bfloat16
        assert get_allgather_dtype(param, param.ds_tensor) == torch.bfloat16


class TestZero3AutocastMixedDtype(DistributedTest):
    world_size = 2

    def test_fp32_adapter_with_bf16_base_params(self):
        if not bf16_required_version_check():
            pytest.skip("BF16 ZeRO-3 autocast test requires BF16 accelerator support.")

        hidden_dim = 8
        config = _zero3_bf16_autocast_config()

        with deepspeed.zero.Init(config_dict_or_path=config):
            model = MixedDtypeAdapterModule(hidden_dim)
        model.attach_fp32_adapter(rank=4)
        _assert_mixed_partition_dtypes(model)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=0.1)
        engine, _, _, _ = deepspeed.initialize(config=config,
                                               model=model,
                                               model_parameters=trainable_params,
                                               optimizer=optimizer)
        try:
            _assert_mixed_partition_dtypes(engine.module)
            _assert_autocast_comm_dtype(engine.module)

            adapter_a_before = safe_get_full_fp32_param(engine.module.adapter_a).detach().clone()
            device = engine.device
            x = torch.randn(2, hidden_dim, device=device, dtype=torch.float32)
            target = torch.randn(2, hidden_dim, device=device, dtype=torch.float32)

            loss = engine(x, target)
            engine.backward(loss)

            adapter_a_grad = safe_get_full_grad(engine.module.adapter_a)
            assert adapter_a_grad is not None
            assert torch.count_nonzero(adapter_a_grad).item() > 0

            engine.step()

            adapter_a_after = safe_get_full_fp32_param(engine.module.adapter_a)
            assert not torch.equal(adapter_a_before, adapter_a_after)
            _assert_mixed_partition_dtypes(engine.module)
        finally:
            engine.destroy()


@contextmanager
def _zero1_engine(autocast):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "bf16": {
            "enabled": True
        },
        "data_types": {
            "grad_accum_dtype": "fp32"
        },
        "communication_data_type": "fp32",
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {
                "device": "cpu"
            },
            "reduce_bucket_size": 64,
        },
        "zero_force_ds_cpu_optimizer": False,
        "torch_autocast": {
            "enabled": autocast,
            "dtype": "bfloat16"
        },
    }
    model = torch.nn.Linear(16, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
    try:
        yield engine
    finally:
        engine.destroy()


def _check_reduced_gradient(engine, expected):
    # The two ranks' inputs are adjacent BF16 values. Their mean is exact in FP32
    # and rounds to 1 in BF16. Each weight gradient equals this mean.
    value = 1 + dist.get_rank() / 128
    inputs = torch.full((1, 16), value, dtype=torch.bfloat16, device=engine.device)
    engine.backward(engine(inputs).sum())
    grad = safe_get_full_grad(engine.module.weight)
    torch.testing.assert_close(grad, torch.full_like(grad, expected, dtype=torch.float32), rtol=0, atol=0)
    engine.step()


@pytest.mark.skipif(not bf16_required_version_check(), reason="BF16 support required")
class TestZero1AutocastEngineIsolation(DistributedTest):
    world_size = 2

    def test_sequential_engines(self):
        with _zero1_engine(autocast=False) as fresh:
            _check_reduced_gradient(fresh, 1 + 1 / 256)
        with _zero1_engine(autocast=True) as autocast:
            _check_reduced_gradient(autocast, 1)
        with _zero1_engine(autocast=False) as plain:
            _check_reduced_gradient(plain, 1 + 1 / 256)

    @pytest.mark.parametrize("autocast_first", [True, False])
    def test_coexisting_engines(self, autocast_first):
        with _zero1_engine(autocast=autocast_first) as first:
            with _zero1_engine(autocast=not autocast_first) as second:
                autocast, plain = (first, second) if autocast_first else (second, first)
                _check_reduced_gradient(autocast, 1)
                _check_reduced_gradient(plain, 1 + 1 / 256)
                _check_reduced_gradient(autocast, 1)
