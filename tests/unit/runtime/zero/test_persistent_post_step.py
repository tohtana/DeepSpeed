# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

from deepspeed.runtime.zero import partition_parameters as partition_module
from deepspeed.runtime.zero import stage3 as stage3_module
from deepspeed.runtime.zero.partition_parameters import Init, ZeroParamStatus
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.utils import nvtx


class _FakeAccelerator:

    def __init__(self, resolves_data_dependency=True):
        self._resolves_data_dependency = resolves_data_dependency
        self.synchronize = MagicMock()

    def device_name(self):
        return "cpu"

    def on_accelerator(self, tensor):
        return True

    def resolves_data_dependency(self):
        return self._resolves_data_dependency


def _partitioned_param(partition_size, dtype=torch.float32):
    param = torch.nn.Parameter(torch.empty(partition_size * 2), requires_grad=False)
    param.ds_tensor = torch.arange(partition_size, dtype=dtype)
    param.ds_tensor.ds_numel = partition_size
    param.ds_numel = partition_size * 2
    param.ds_shape = torch.Size([partition_size * 2])
    return param


def _init_for_coalesced_gather(monkeypatch, use_all_gather_into_tensor=True, resolves_data_dependency=True):
    accelerator = _FakeAccelerator(resolves_data_dependency=resolves_data_dependency)
    monkeypatch.setattr(partition_module, "get_accelerator", lambda: accelerator)
    zero_init = object.__new__(Init)
    zero_init.local_device = torch.device("cpu")
    zero_init.use_all_gather_into_tensor = use_all_gather_into_tensor
    zero_init.get_partition_dp_group = MagicMock(return_value=object())
    zero_init._partition_world_size = MagicMock(return_value=2)
    return zero_init, accelerator


def _fill_all_gather_output(output, input_tensor):
    output.copy_(torch.cat([input_tensor, input_tensor + 10]))


def test_eligible_multi_parameter_gather_uses_one_work(monkeypatch):
    zero_init, accelerator = _init_for_coalesced_gather(monkeypatch)
    params = [_partitioned_param(2), _partitioned_param(3)]
    work = MagicMock()
    captured_outputs = []

    def submit(outputs, inputs, group):
        captured_outputs.extend(outputs)
        work.wait.side_effect = lambda: [
            _fill_all_gather_output(output, input_tensor) for output, input_tensor in zip(outputs, inputs)
        ]
        return work

    submit_mock = MagicMock(side_effect=submit)
    individual_mock = MagicMock()
    monkeypatch.setattr(partition_module.dist, "try_all_gather_into_tensor_coalesced", submit_mock)
    monkeypatch.setattr(partition_module.dist, "all_gather_into_tensor", individual_mock)

    zero_init._allgather_params_coalesced(params, use_post_step_coalesced_fast_path=True)

    submit_mock.assert_called_once()
    individual_mock.assert_not_called()
    work.wait.assert_called_once_with()
    assert params[0].data.data_ptr() == captured_outputs[0].data_ptr()
    assert params[1].data.data_ptr() == captured_outputs[1].data_ptr()
    assert torch.equal(params[0], torch.tensor([0.0, 1.0, 10.0, 11.0]))
    assert torch.equal(params[1], torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]))
    accelerator.synchronize.assert_not_called()


@pytest.mark.parametrize("resolves_data_dependency", [True, False])
def test_none_from_adapter_runs_exact_individual_loop(monkeypatch, resolves_data_dependency):
    zero_init, accelerator = _init_for_coalesced_gather(monkeypatch, resolves_data_dependency=resolves_data_dependency)
    params = [_partitioned_param(2), _partitioned_param(3)]
    works = [MagicMock(), MagicMock()]
    individual_mock = MagicMock(side_effect=lambda output, input_tensor, **_kwargs:
                                (_fill_all_gather_output(output, input_tensor), works.pop(0))[1])
    submit_mock = MagicMock(return_value=None)
    monkeypatch.setattr(partition_module.dist, "try_all_gather_into_tensor_coalesced", submit_mock)
    monkeypatch.setattr(partition_module.dist, "all_gather_into_tensor", individual_mock)

    zero_init._allgather_params_coalesced(params, use_post_step_coalesced_fast_path=True)

    submit_mock.assert_called_once()
    assert individual_mock.call_count == 2
    assert torch.equal(params[0], torch.tensor([0.0, 1.0, 10.0, 11.0]))
    assert torch.equal(params[1], torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]))
    if resolves_data_dependency:
        accelerator.synchronize.assert_not_called()
    else:
        accelerator.synchronize.assert_called_once_with()


def test_submitted_exception_does_not_run_individual_loop(monkeypatch):
    zero_init, _ = _init_for_coalesced_gather(monkeypatch)
    params = [_partitioned_param(2), _partitioned_param(3)]
    submit_mock = MagicMock(side_effect=RuntimeError("collective failed"))
    individual_mock = MagicMock()
    monkeypatch.setattr(partition_module.dist, "try_all_gather_into_tensor_coalesced", submit_mock)
    monkeypatch.setattr(partition_module.dist, "all_gather_into_tensor", individual_mock)

    with pytest.raises(RuntimeError, match="collective failed"):
        zero_init._allgather_params_coalesced(params, use_post_step_coalesced_fast_path=True)
    individual_mock.assert_not_called()


def test_quantized_gather_never_calls_adapter(monkeypatch):
    zero_init, _ = _init_for_coalesced_gather(monkeypatch)
    params = [_partitioned_param(2, dtype=torch.int8), _partitioned_param(3, dtype=torch.int8)]
    for param in params:
        param.ds_tensor.ds_quant_scale = torch.ones(1)
    zero_init.quantizer_module = MagicMock()
    zero_init.quantizer_module.dequantize.side_effect = lambda gathered, _scales: gathered.float()
    works = []

    def individual(output, input_tensor, **_kwargs):
        _fill_all_gather_output(output, input_tensor)
        work = MagicMock()
        works.append(work)
        return work

    submit_mock = MagicMock()
    individual_mock = MagicMock(side_effect=individual)
    monkeypatch.setattr(partition_module.dist, "try_all_gather_into_tensor_coalesced", submit_mock)
    monkeypatch.setattr(partition_module.dist, "all_gather_into_tensor", individual_mock)

    zero_init._allgather_params_coalesced(params, quantize=True, use_post_step_coalesced_fast_path=True)

    submit_mock.assert_not_called()
    assert individual_mock.call_count == 4
    works[1].wait.assert_called_once_with()
    works[3].wait.assert_called_once_with()


def test_world_size_one_returns_before_adapter(monkeypatch):
    zero_init, _ = _init_for_coalesced_gather(monkeypatch)
    zero_init._partition_world_size.return_value = 1
    params = [_partitioned_param(2), _partitioned_param(3)]
    no_gather_work = MagicMock()
    no_gather_mock = MagicMock(return_value=no_gather_work)
    submit_mock = MagicMock()
    monkeypatch.setattr(partition_module, "_no_gather_coalesced", no_gather_mock)
    monkeypatch.setattr(partition_module.dist, "try_all_gather_into_tensor_coalesced", submit_mock)

    zero_init._allgather_params_coalesced(params, use_post_step_coalesced_fast_path=True)

    no_gather_mock.assert_called_once_with(params)
    no_gather_work.wait.assert_called_once_with()
    submit_mock.assert_not_called()


def _init_for_routing():
    zero_init = object.__new__(Init)
    zero_init._ensure_availability_of_partitioned_params = MagicMock()
    zero_init.get_partition_dp_group = MagicMock(return_value=object())
    zero_init.allgather_sequential = False
    zero_init._allgather_params_sequential = MagicMock()
    zero_init._allgather_params_coalesced = MagicMock()
    return zero_init


def _unavailable_param():
    return SimpleNamespace(ds_status=ZeroParamStatus.NOT_AVAILABLE, ds_tensor=torch.ones(1))


def test_singleton_uses_existing_sequential_path(monkeypatch):
    monkeypatch.setattr(nvtx, "enable_nvtx", False)
    zero_init = _init_for_routing()
    param = _unavailable_param()

    zero_init._all_gather([param], use_post_step_coalesced_fast_path=True)

    zero_init._allgather_params_sequential.assert_called_once_with([param], hierarchy=None)
    zero_init._allgather_params_coalesced.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_routing_flag_reaches_only_coalesced_buckets(monkeypatch, enabled):
    monkeypatch.setattr(nvtx, "enable_nvtx", False)
    zero_init = _init_for_routing()
    params = [_unavailable_param(), _unavailable_param()]

    zero_init._all_gather(params, use_post_step_coalesced_fast_path=enabled)

    assert zero_init._allgather_params_coalesced.call_args_list == [
        call(params, None, quantize=False, use_post_step_coalesced_fast_path=enabled),
        call([], None, quantize=True, use_post_step_coalesced_fast_path=enabled),
    ]


def _stage3_optimizer(persistent_parameters):
    optimizer = object.__new__(DeepSpeedZeroOptimizer_Stage3)
    optimizer.offload_optimizer = False
    optimizer.persistent_parameters = persistent_parameters
    optimizer.swap_optimizer = False
    optimizer.invalidate_secondary_tensor = MagicMock()
    optimizer.timers = MagicMock()
    return optimizer


def test_post_step_is_the_only_true_opt_in(monkeypatch):
    monkeypatch.setattr(nvtx, "enable_nvtx", False)
    monkeypatch.setattr(stage3_module, "see_memory_usage", MagicMock())
    monkeypatch.setattr(stage3_module, "print_rank_0", MagicMock())
    first = SimpleNamespace(all_gather=MagicMock())
    params = [first, object()]
    optimizer = _stage3_optimizer(params)

    optimizer._post_step([])

    first.all_gather.assert_called_once_with(params, use_post_step_coalesced_fast_path=True)
    optimizer.invalidate_secondary_tensor.assert_called_once_with()


def test_empty_post_step_submits_no_gather(monkeypatch):
    monkeypatch.setattr(nvtx, "enable_nvtx", False)
    monkeypatch.setattr(stage3_module, "see_memory_usage", MagicMock())
    monkeypatch.setattr(stage3_module, "print_rank_0", MagicMock())
    optimizer = _stage3_optimizer([])

    optimizer._post_step([])

    optimizer.invalidate_secondary_tensor.assert_called_once_with()


def test_checkpoint_event_keeps_default_gather_call():
    first = SimpleNamespace(all_gather=MagicMock())
    params = [first, object()]
    optimizer = _stage3_optimizer(params)

    optimizer.checkpoint_event_epilogue()

    first.all_gather.assert_called_once_with(params)
