# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Regression: ZeRO-3 patched F.linear must work with torch.func transforms.

After deepspeed.initialize with ZeRO Stage 3, ``torch.nn.functional.linear`` is
replaced with ``LinearFunctionForZeroStage3``. That autograd.Function must use
the ``forward`` + ``setup_context`` pattern (PyTorch 2.0+); the legacy
``forward(ctx, ...)`` + ``ctx.save_for_backward`` in forward raises::

    RuntimeError: In order to use an autograd.Function with functorch
    transforms ... it must override the setup_context staticmethod.

See ``repro_zero3_functorch_linear.py`` for a standalone script version.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import deepspeed
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest


def _zero3_functorch_config():
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 2147483647,
        "zero_optimization": {
            "stage": 3,
            "stage3_param_persistence_threshold": 0,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            },
        },
    }
    acc = get_accelerator()
    if acc.is_bf16_supported():
        config["bf16"] = {"enabled": True}
    elif acc.is_fp16_supported():
        config["fp16"] = {"enabled": True, "initial_scale_power": 8}
    return config


class TestZeroFunctorchLinearRegression(DistributedTest):
    """``torch.func.grad_and_value`` over ZeRO-3 memory-efficient F.linear."""

    world_size = 1

    def test_grad_and_value_over_patched_functional_linear(self):
        if not hasattr(torch, "func"):
            pytest.skip("torch.func not available")
        if not hasattr(torch.autograd.Function, "setup_context"):
            pytest.skip("Requires PyTorch 2.0+ autograd.Function.setup_context")

        model = nn.Linear(8, 8, bias=True)
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=_zero3_functorch_config(),
            model_parameters=model.parameters(),
        )

        device = engine.device
        dtype = engine.module.weight.dtype
        weight = torch.randn(8, 8, device=device, dtype=dtype, requires_grad=True)
        inp = torch.randn(2, 8, device=device, dtype=dtype, requires_grad=True)

        def loss_fn(w, x):
            return F.linear(x, w, None).sum()

        grads, value = torch.func.grad_and_value(loss_fn, argnums=(0, 1))(weight, inp)
        assert torch.isfinite(value)
        assert grads[0] is not None and torch.isfinite(grads[0]).all()
        assert grads[1] is not None and torch.isfinite(grads[1]).all()
