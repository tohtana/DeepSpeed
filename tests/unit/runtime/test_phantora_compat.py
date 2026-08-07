# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math

import pytest
import torch

import deepspeed.runtime.bf16_optimizer as bf16_optimizer
import deepspeed.runtime.zero.utils as zero_utils
import deepspeed.utils.nvtx as nvtx


class MockGloo:

    def __init__(self, rank0_length, rank0_values):
        self.rank0_length = rank0_length
        self.rank0_values = rank0_values
        self.group = object()
        self.new_group_calls = 0
        self.broadcast_calls = []

    def new_group(self, *, backend):
        assert backend == "gloo"
        self.new_group_calls += 1
        return self.group

    def broadcast(self, tensor, *, src, group, async_op):
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.int64
        assert src == 0
        assert group is self.group
        assert async_op is False
        self.broadcast_calls.append(tensor.numel())
        if tensor.numel() == 1:
            tensor.fill_(self.rank0_length)
        else:
            tensor.copy_(torch.tensor(self.rank0_values, dtype=torch.int64))


@pytest.fixture(autouse=True)
def reset_compat_state(monkeypatch):
    monkeypatch.delenv("PHANTORA", raising=False)
    monkeypatch.setattr(nvtx, "enable_nvtx", False)
    monkeypatch.setattr(zero_utils, "_phantora_gloo_group", None)


@pytest.mark.parametrize("value", [None, "0", "true", "01"])
def test_zero3_validation_uses_native_path_unless_exactly_enabled(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("PHANTORA", value)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(zero_utils.dist, "get_rank", lambda: 0)
    native_calls = []
    monkeypatch.setattr(zero_utils.dist, "broadcast", lambda *args, **kwargs: native_calls.append((args, kwargs)))
    monkeypatch.setattr(zero_utils, "get_accelerator", lambda: type("Accelerator", (), {
        "device_name": lambda self, _rank: "cpu"
    })())
    monkeypatch.setattr(zero_utils.torch_dist, "new_group",
                        lambda **_kwargs: pytest.fail("Phantora Gloo group must stay disabled"))

    zero_utils.assert_lst_len_same_as_other_ranks([1])
    assert zero_utils.get_lst_from_rank0([1, 2]) == [1, 2]

    assert len(native_calls) == 2
    assert all("group" not in kwargs for _, kwargs in native_calls)


def test_zero3_validation_uses_cached_cpu_gloo_and_keeps_mismatch_checks(monkeypatch):
    monkeypatch.setenv("PHANTORA", "1")
    monkeypatch.setattr(zero_utils.dist, "get_rank", lambda: 1)
    gloo = MockGloo(rank0_length=3, rank0_values=[10, 20, 30])
    monkeypatch.setattr(zero_utils, "torch_dist", gloo)
    monkeypatch.setattr(zero_utils.dist, "broadcast",
                        lambda *_args, **_kwargs: pytest.fail("Phantora must not use the native broadcast"))
    monkeypatch.setattr(zero_utils, "get_accelerator",
                        lambda: pytest.fail("Phantora validation tensors must stay on CPU"))

    zero_utils.assert_lst_len_same_as_other_ranks([-1, -2, -3])
    assert zero_utils.get_lst_from_rank0([-1, -1, -1]) == [10, 20, 30]
    with pytest.raises(RuntimeError, match="list contents"):
        zero_utils.assert_ints_same_as_other_ranks([10, 99, 30])

    assert gloo.new_group_calls == 1
    assert gloo.broadcast_calls == [1, 3, 1, 3]

    gloo.rank0_length = 0
    with pytest.raises(RuntimeError, match="rank0: 0"):
        zero_utils.assert_lst_len_same_as_other_ranks([1])


@pytest.mark.parametrize("helper_name,native_name", [
    ("get_global_norm_of_tensors", "_native_get_global_norm_of_tensors"),
    ("get_norm_with_moe_layers", "_native_get_norm_with_moe_layers"),
])
@pytest.mark.parametrize("invalid_norm", [0.0, -2.0, float("nan"), float("inf"), -float("inf")])
def test_bf16_norm_helpers_use_sentinel_only_for_invalid_phantora_values(monkeypatch, helper_name, native_name,
                                                                         invalid_norm):
    calls = []

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return invalid_norm

    monkeypatch.setattr(bf16_optimizer, native_name, native)
    monkeypatch.setenv("PHANTORA", "1")

    assert getattr(bf16_optimizer, helper_name)("arg", key="value") == 1.0
    assert calls == [(('arg', ), {"key": "value"})]

    monkeypatch.setenv("PHANTORA", "true")
    native_result = getattr(bf16_optimizer, helper_name)()
    if math.isnan(invalid_norm):
        assert math.isnan(native_result)
    else:
        assert native_result == invalid_norm


@pytest.mark.parametrize("helper_name,native_name", [
    ("get_global_norm_of_tensors", "_native_get_global_norm_of_tensors"),
    ("get_norm_with_moe_layers", "_native_get_norm_with_moe_layers"),
])
def test_bf16_norm_helpers_preserve_positive_native_object(monkeypatch, helper_name, native_name):
    positive_norm = torch.tensor(2.0)
    monkeypatch.setattr(bf16_optimizer, native_name, lambda: positive_norm)
    monkeypatch.setenv("PHANTORA", "1")

    assert getattr(bf16_optimizer, helper_name)() is positive_norm
