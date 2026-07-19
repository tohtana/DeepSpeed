# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from unittest.mock import MagicMock

import pytest
import torch
from torch import distributed as torch_dist

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.comm import comm as comm_module
from deepspeed.comm.torch import TorchBackend
from unit.common import DistributedTest


def _backend(group, backend=torch_dist.Backend.NCCL):
    torch_backend = object.__new__(TorchBackend)
    torch_backend.get_world_group = MagicMock(return_value=group)
    torch_backend.get_backend = MagicMock(return_value=backend)
    return torch_backend


def _group(canonical=True, compatibility=True, work=None):
    group = MagicMock()
    if work is None:
        work = MagicMock()
    if canonical:
        group.all_gather_single_coalesced = MagicMock(return_value=work)
    else:
        del group.all_gather_single_coalesced
    if compatibility:
        group.allgather_into_tensor_coalesced = MagicMock(return_value=work)
    else:
        del group.allgather_into_tensor_coalesced
    return group, work


def _tensor_lists():
    inputs = [torch.arange(3, dtype=torch.bfloat16), torch.arange(5, dtype=torch.float32)]
    outputs = [torch.empty(6, dtype=torch.bfloat16), torch.empty(10, dtype=torch.float32)]
    return outputs, inputs


def test_canonical_name_precedes_compatibility_name(monkeypatch):
    monkeypatch.setattr(torch_dist, "get_debug_level", lambda: torch_dist.DebugLevel.OFF)
    group, work = _group()
    torch_backend = _backend(group)
    outputs, inputs = _tensor_lists()

    result = torch_backend.try_all_gather_into_tensor_coalesced(outputs, inputs)

    assert result is work
    torch_backend.get_world_group.assert_called_once_with()
    torch_backend.get_backend.assert_called_once_with(group)
    group.all_gather_single_coalesced.assert_called_once()
    group.allgather_into_tensor_coalesced.assert_not_called()
    call_outputs, call_inputs, opts = group.all_gather_single_coalesced.call_args.args
    assert call_outputs is outputs
    assert call_inputs is inputs
    assert opts.asyncOp is True


def test_compatibility_name_is_selected_when_canonical_name_is_absent(monkeypatch):
    monkeypatch.setattr(torch_dist, "get_debug_level", lambda: torch_dist.DebugLevel.INFO)
    group, work = _group(canonical=False)
    torch_backend = _backend(group)
    outputs, inputs = _tensor_lists()

    result = torch_backend.try_all_gather_into_tensor_coalesced(outputs, inputs, group=group)

    assert result is work
    torch_backend.get_world_group.assert_not_called()
    group.allgather_into_tensor_coalesced.assert_called_once()
    call_outputs, call_inputs, opts = group.allgather_into_tensor_coalesced.call_args.args
    assert call_outputs is outputs
    assert call_inputs is inputs
    assert opts.asyncOp is True


def test_detail_falls_back_before_group_or_backend_resolution(monkeypatch):
    monkeypatch.setattr(torch_dist, "get_debug_level", lambda: torch_dist.DebugLevel.DETAIL)
    group, _ = _group()
    torch_backend = _backend(group)
    torch_backend.get_world_group.side_effect = AssertionError("DETAIL must be the first guard")

    assert torch_backend.try_all_gather_into_tensor_coalesced([], []) is None
    torch_backend.get_backend.assert_not_called()
    group.all_gather_single_coalesced.assert_not_called()
    group.allgather_into_tensor_coalesced.assert_not_called()


def test_non_nccl_backend_falls_back_before_method_lookup(monkeypatch):
    monkeypatch.setattr(torch_dist, "get_debug_level", lambda: torch_dist.DebugLevel.OFF)
    group, _ = _group()
    torch_backend = _backend(group, backend=torch_dist.Backend.GLOO)
    outputs, inputs = _tensor_lists()

    assert torch_backend.try_all_gather_into_tensor_coalesced(outputs, inputs, group=group) is None
    group.all_gather_single_coalesced.assert_not_called()
    group.allgather_into_tensor_coalesced.assert_not_called()


def test_absent_methods_fall_back_without_submission(monkeypatch):
    monkeypatch.setattr(torch_dist, "get_debug_level", lambda: torch_dist.DebugLevel.OFF)
    group, _ = _group(canonical=False, compatibility=False)
    torch_backend = _backend(group)
    outputs, inputs = _tensor_lists()

    assert torch_backend.try_all_gather_into_tensor_coalesced(outputs, inputs, group=group) is None


@pytest.mark.parametrize("outputs,inputs", [([], []), ([torch.empty(2)], [])])
def test_invalid_tensor_lists_fail_before_submission(monkeypatch, outputs, inputs):
    monkeypatch.setattr(torch_dist, "get_debug_level", lambda: torch_dist.DebugLevel.OFF)
    group, _ = _group()
    torch_backend = _backend(group)

    with pytest.raises(ValueError, match="equal non-empty"):
        torch_backend.try_all_gather_into_tensor_coalesced(outputs, inputs, group=group)
    group.all_gather_single_coalesced.assert_not_called()
    group.allgather_into_tensor_coalesced.assert_not_called()


def test_submitted_exception_propagates_without_compatibility_retry(monkeypatch):
    monkeypatch.setattr(torch_dist, "get_debug_level", lambda: torch_dist.DebugLevel.OFF)
    group, _ = _group()
    group.all_gather_single_coalesced.side_effect = RuntimeError("collective failed")
    torch_backend = _backend(group)
    outputs, inputs = _tensor_lists()

    with pytest.raises(RuntimeError, match="collective failed"):
        torch_backend.try_all_gather_into_tensor_coalesced(outputs, inputs, group=group)
    group.all_gather_single_coalesced.assert_called_once()
    group.allgather_into_tensor_coalesced.assert_not_called()


@pytest.mark.parametrize("backend_state", ["missing", "uninitialized"])
def test_comm_dispatcher_falls_back_for_unsupported_backend(monkeypatch, backend_state):
    backend = MagicMock()
    if backend_state == "missing":
        del backend.try_all_gather_into_tensor_coalesced
        backend.is_initialized.return_value = True
    else:
        backend.is_initialized.return_value = False
    monkeypatch.setattr(comm_module, "cdb", backend)

    assert comm_module.try_all_gather_into_tensor_coalesced([], []) is None


def test_comm_dispatcher_forwards_original_lists(monkeypatch):
    outputs, inputs = _tensor_lists()
    group = object()
    work = object()
    backend = MagicMock()
    backend.is_initialized.return_value = True
    backend.try_all_gather_into_tensor_coalesced.return_value = work
    monkeypatch.setattr(comm_module, "cdb", backend)

    result = comm_module.try_all_gather_into_tensor_coalesced(outputs, inputs, group=group)

    assert result is work
    backend.try_all_gather_into_tensor_coalesced.assert_called_once_with(output_tensors=outputs,
                                                                         input_tensors=inputs,
                                                                         group=group)


class TestAllGatherSingleCoalescedNCCL(DistributedTest):
    world_size = 2

    def test_unequal_disjoint_outputs(self):
        assert torch_dist.get_debug_level() != torch_dist.DebugLevel.DETAIL
        device = get_accelerator().current_device_name()
        rank = dist.get_rank()
        inputs = [
            torch.arange(3, device=device, dtype=torch.bfloat16) + rank * 10,
            torch.arange(5, device=device, dtype=torch.float32) + rank * 100,
        ]
        outputs = [
            torch.empty(6, device=device, dtype=torch.bfloat16),
            torch.empty(10, device=device, dtype=torch.float32),
        ]

        work = dist.try_all_gather_into_tensor_coalesced(outputs, inputs, group=dist.get_world_group())

        assert work is not None
        work.wait()
        assert torch.equal(
            outputs[0],
            torch.cat([
                torch.arange(3, device=device, dtype=torch.bfloat16),
                torch.arange(3, device=device, dtype=torch.bfloat16) + 10
            ]))
        assert torch.equal(
            outputs[1],
            torch.cat([
                torch.arange(5, device=device, dtype=torch.float32),
                torch.arange(5, device=device, dtype=torch.float32) + 100
            ]))


class TestAllGatherSingleCoalescedDetailNCCL(DistributedTest):
    world_size = 2

    def test_individual_fallback(self):
        assert torch_dist.get_debug_level() == torch_dist.DebugLevel.DETAIL
        device = get_accelerator().current_device_name()
        rank = dist.get_rank()
        inputs = [
            torch.arange(3, device=device, dtype=torch.bfloat16) + rank * 10,
            torch.arange(5, device=device, dtype=torch.bfloat16) + rank * 100,
        ]
        outputs = [
            torch.empty(6, device=device, dtype=torch.bfloat16),
            torch.empty(10, device=device, dtype=torch.bfloat16),
        ]

        assert dist.try_all_gather_into_tensor_coalesced(outputs, inputs, group=dist.get_world_group()) is None
        works = [
            dist.all_gather_into_tensor(output, input_tensor, async_op=True)
            for output, input_tensor in zip(outputs, inputs)
        ]
        works[-1].wait()
        assert torch.equal(
            outputs[0],
            torch.cat([
                torch.arange(3, device=device, dtype=torch.bfloat16),
                torch.arange(3, device=device, dtype=torch.bfloat16) + 10
            ]))
        assert torch.equal(
            outputs[1],
            torch.cat([
                torch.arange(5, device=device, dtype=torch.bfloat16),
                torch.arange(5, device=device, dtype=torch.bfloat16) + 100
            ]))
