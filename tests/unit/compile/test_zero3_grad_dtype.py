# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.compile.init_z3 import _allow_dynamo_dynamic_parameter_shapes_for_z3, _resolve_expected_grad_dtype


def test_missing_grad_dtype_attribute_falls_back_to_param_dtype():

    class FakeParam:
        dtype = torch.bfloat16

    assert _resolve_expected_grad_dtype(FakeParam()) is torch.bfloat16


def test_explicit_none_grad_dtype_allows_raw_grad_dtype():
    param = torch.empty((2, 3), dtype=torch.bfloat16)
    param.grad_dtype = None

    assert _resolve_expected_grad_dtype(param) is None


def test_explicit_grad_dtype_is_preserved():
    param = torch.empty((2, 3), dtype=torch.bfloat16)
    param.grad_dtype = torch.float32

    assert _resolve_expected_grad_dtype(param) is torch.float32


def test_zero3_allows_dynamo_dynamic_parameter_shapes(monkeypatch):

    class FakeDynamoConfig:
        force_parameter_static_shapes = True
        force_nn_module_property_static_shapes = True

    class FakeDynamo:
        config = FakeDynamoConfig()

    compile_kwargs = {}
    monkeypatch.setattr(torch, "_dynamo", FakeDynamo)

    assert _allow_dynamo_dynamic_parameter_shapes_for_z3(compile_kwargs)
    assert "dynamic" not in compile_kwargs
    assert FakeDynamo.config.force_parameter_static_shapes is False
    assert FakeDynamo.config.force_nn_module_property_static_shapes is False


def test_zero3_preserves_explicit_dynamo_dynamic_setting(monkeypatch):

    class FakeDynamoConfig:
        force_parameter_static_shapes = True
        force_nn_module_property_static_shapes = True

    class FakeDynamo:
        config = FakeDynamoConfig()

    compile_kwargs = {"dynamic": False}
    monkeypatch.setattr(torch, "_dynamo", FakeDynamo)

    assert _allow_dynamo_dynamic_parameter_shapes_for_z3(compile_kwargs)
    assert compile_kwargs["dynamic"] is False
