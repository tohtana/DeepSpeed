# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools
from types import SimpleNamespace

import pytest
import torch
import deepspeed.runtime.zero.linear as zero_linear
from deepspeed.runtime.zero.linear import LinearModuleForZeroStage3
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest


@pytest.mark.parametrize('half_op', [False, True])
class TestAutoCastDisable(DistributedTest):

    def test_missing_amp_autocast(self, half_op):
        hidden_dim = 4
        if half_op:
            input = torch.randn(hidden_dim).to(get_accelerator().device_name()).half()
            ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).to(get_accelerator().device_name()).half()
        else:
            input = torch.randn(hidden_dim).to(get_accelerator().device_name())
            ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).to(get_accelerator().device_name())

        output = ds_linear(input)
        assert output.dtype == ds_linear.weight.dtype

    def test_disable_autocast_linear(self, half_op):
        hidden_dim = 4
        if half_op:
            input = torch.randn(hidden_dim).to(get_accelerator().device_name()).half()
            ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).to(get_accelerator().device_name()).half()
        else:
            input = torch.randn(hidden_dim).to(get_accelerator().device_name())
            ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).to(get_accelerator().device_name())

        with torch.amp.autocast(device_type=get_accelerator().device_name(), enabled=False):
            output = ds_linear(input)
            assert output.dtype == ds_linear.weight.dtype


@pytest.mark.parametrize('half_input, half_weight', [(False, False), (False, True), (True, False), (True, True)])
class TestAutoCastEnable(DistributedTest):

    def test_autocast_linear(self, tmpdir, half_input, half_weight):
        hidden_dim = 4
        input = torch.randn(hidden_dim).to(get_accelerator().device_name())
        ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).to(get_accelerator().device_name())

        if half_input:
            input = input.half()

        if half_weight:
            ds_linear = ds_linear.half()

        with torch.amp.autocast(device_type=get_accelerator().device_name()):
            output = ds_linear(input)
            assert output.dtype == torch.half or output.dtype == torch.bfloat16


class _FakeAccelerator:

    def __init__(self, device_type):
        self._device_type = device_type

    def device_name(self):
        return self._device_type


def test_get_autocast_decorators_prefers_torch_amp(monkeypatch):

    def custom_fwd(*args, **kwargs):
        return None

    def custom_bwd(*args, **kwargs):
        return None

    fake_torch = SimpleNamespace(amp=SimpleNamespace(custom_fwd=custom_fwd, custom_bwd=custom_bwd))
    monkeypatch.setattr(zero_linear, 'torch', fake_torch)
    monkeypatch.setattr(zero_linear, 'get_accelerator', lambda: _FakeAccelerator('cuda'))

    autocast_custom_fwd, autocast_custom_bwd = zero_linear._get_autocast_decorators()

    assert isinstance(autocast_custom_fwd, functools.partial)
    assert isinstance(autocast_custom_bwd, functools.partial)
    assert autocast_custom_fwd.func is custom_fwd
    assert autocast_custom_bwd.func is custom_bwd
    assert autocast_custom_fwd.keywords == {'device_type': 'cuda'}
    assert autocast_custom_bwd.keywords == {'device_type': 'cuda'}


def test_get_autocast_decorators_uses_legacy_backend_amp(monkeypatch):

    def custom_fwd(*args, **kwargs):
        return None

    def custom_bwd(*args, **kwargs):
        return None

    fake_torch = SimpleNamespace(
        amp=SimpleNamespace(), npu=SimpleNamespace(amp=SimpleNamespace(custom_fwd=custom_fwd, custom_bwd=custom_bwd)))
    monkeypatch.setattr(zero_linear, 'torch', fake_torch)
    monkeypatch.setattr(zero_linear, 'get_accelerator', lambda: _FakeAccelerator('npu'))

    autocast_custom_fwd, autocast_custom_bwd = zero_linear._get_autocast_decorators()

    assert autocast_custom_fwd is custom_fwd
    assert autocast_custom_bwd is custom_bwd


def test_get_autocast_decorators_falls_back_to_noop(monkeypatch):
    fake_torch = SimpleNamespace(amp=SimpleNamespace())
    monkeypatch.setattr(zero_linear, 'torch', fake_torch)
    monkeypatch.setattr(zero_linear, 'get_accelerator', lambda: _FakeAccelerator('hpu'))

    autocast_custom_fwd, autocast_custom_bwd = zero_linear._get_autocast_decorators()

    assert autocast_custom_fwd is zero_linear.noop_decorator
    assert autocast_custom_bwd is zero_linear.noop_decorator
