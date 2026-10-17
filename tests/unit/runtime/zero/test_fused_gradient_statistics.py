# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import gc
from types import SimpleNamespace
import weakref
from unittest import mock

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
import deepspeed.runtime.zero.fused_gradient_statistics as fused_stats
from deepspeed.runtime.zero.fused_gradient_statistics import (FusedGradientStatistics, build_gradient_statistics_plan)
from unit.common import DistributedTest
from unit.simple_model import SimpleModel


def _parameter(**attributes):
    return SimpleNamespace(**attributes)


def _reference_statistics(plan, stable_descriptors=None):
    sum_squares = []
    invalid = []
    for gradient in plan.gradients:
        flat = gradient.detach().reshape(-1)
        sum_squares.append(flat.double().square().sum())
        invalid.append((~torch.isfinite(flat)).any())
    if sum_squares:
        sum_squares = torch.stack(sum_squares)
        invalid = torch.stack(invalid)
    else:
        sum_squares = torch.empty(0, dtype=torch.float64)
        invalid = torch.empty(0, dtype=torch.bool)
    descriptors = stable_descriptors or (("test", plan.lengths), )
    return sum_squares, invalid, (None, descriptors, None, None)


def test_plan_preserves_selection_and_checks_all_gradients():
    regular = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    padding = torch.tensor([float("inf")], dtype=torch.bfloat16)
    padding._zero_padding = True
    model_parallel = torch.arange(7, dtype=torch.bfloat16)
    empty = torch.empty(0, dtype=torch.bfloat16)
    replica = torch.tensor([float("nan")], dtype=torch.bfloat16)
    gradient_groups = [[regular, padding, model_parallel], [empty, replica]]
    parameter_groups = [[_parameter(), _parameter(tensor_model_parallel=True)],
                        [_parameter(model_parallel=True),
                         _parameter(ds_pipe_replicated=True)]]

    plan = build_gradient_statistics_plan(gradient_groups, parameter_groups, model_parallel_rank=1, chunk_size=3)
    sum_squares, invalid, _ = _reference_statistics(plan)

    assert plan.group_selected == ((2, ), (3, ))
    assert plan.chunk_counts == (1, 1, 3, 0, 1)
    assert sum_squares[0].item() == 25.0
    assert invalid.tolist() == [False, True, False, False, True]


def test_runtime_fallback_is_atomic_on_unsupported_gradients():
    helper = FusedGradientStatistics(model_parallel_rank=0)
    gradients = [[torch.ones(4, dtype=torch.bfloat16)]]
    parameters = [[_parameter()]]

    assert helper.refresh(gradients, parameters) is None
    assert "CUDA" in helper.last_fallback_reason
    assert not helper.active_for_update
    assert helper.cache is None


def test_cache_refreshes_same_storage_without_retaining_gradients():
    gradient = torch.ones(8, dtype=torch.bfloat16)
    gradient_reference = weakref.ref(gradient)
    gradients = [[gradient]]
    parameters = [[_parameter()]]
    helper = FusedGradientStatistics(model_parallel_rank=0)

    with mock.patch.object(fused_stats, "_preflight_plan",
                           return_value=None), mock.patch.object(fused_stats,
                                                                 "_run_fused_gradient_statistics",
                                                                 side_effect=_reference_statistics):
        assert not helper.refresh(gradients, parameters).item()
        first = helper.local_group_sum_squares(gradients[0], parameters[0]).sqrt()
        gradient.mul_(2)
        assert not helper.refresh(gradients, parameters).item()
        second = helper.local_group_sum_squares(gradients[0], parameters[0]).sqrt()

    assert second.item() == 2 * first.item()
    assert helper.refresh_count == 2
    assert helper.cache.plan.gradients == ()
    gradients.clear()
    del gradient
    gc.collect()
    assert gradient_reference() is None


def _assert_cuda_statistics(plan, sum_squares, invalid):
    reference_sum_squares, reference_invalid, _ = _reference_statistics(plan)
    reference_sum_squares = reference_sum_squares.to(sum_squares.device)
    reference_invalid = reference_invalid.to(invalid.device)
    assert torch.equal(invalid, reference_invalid)
    finite = torch.isfinite(reference_sum_squares)
    relative_error = ((sum_squares[finite] - reference_sum_squares[finite]).abs() /
                      reference_sum_squares[finite].abs().clamp_min(torch.finfo(torch.float64).tiny))
    maximum = relative_error.max().item() if relative_error.numel() else 0.0
    assert maximum <= 1e-12


@pytest.mark.skipif(not get_accelerator().is_available(), reason="an accelerator is required")
def test_cuda_kernel_matches_fp64_and_detects_nonfinite_values():
    if not FusedGradientStatistics.is_available():
        pytest.skip("Triton is unavailable")

    finfo = torch.finfo(torch.bfloat16)
    values = torch.tensor([0.0, finfo.tiny, -finfo.tiny, finfo.max, -finfo.max, finfo.tiny / 128, 1.0, -1.0],
                          dtype=torch.bfloat16,
                          device="cuda")
    excluded = torch.zeros(4097, dtype=torch.bfloat16, device="cuda")
    parameters = [[_parameter(), _parameter(ds_pipe_replicated=True)]]
    plan = build_gradient_statistics_plan([[values, excluded]], parameters, model_parallel_rank=0)

    sum_squares, invalid, descriptors = fused_stats._run_fused_gradient_statistics(plan)
    _assert_cuda_statistics(plan, sum_squares, invalid)
    for nonfinite in (float("nan"), float("inf"), -float("inf")):
        excluded[0] = nonfinite
        sum_squares, invalid, descriptors = fused_stats._run_fused_gradient_statistics(plan, descriptors[1])
        assert invalid.tolist() == [False, True]
    excluded.zero_()


def _integration_config(fused):
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_clipping": 0.5,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-3,
            },
        },
        "bf16": {
            "enabled": True,
            "check_grad_overflow": True,
            "bf16_master_weights_and_grads": True,
            "bf16_optimizer_states": True,
        },
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "fused_gradient_statistics": fused,
        },
    }


class TestFusedGradientStatisticsIntegration(DistributedTest):
    world_size = 2

    def test_real_zero2_native_parity_and_cross_rank_overflow_skip(self):
        if not get_accelerator().is_bf16_supported() or not FusedGradientStatistics.is_available():
            pytest.skip("BF16 CUDA with Triton is required")

        torch.manual_seed(1234)
        native_model = SimpleModel(hidden_dim=8)
        fused_model = SimpleModel(hidden_dim=8)
        fused_model.load_state_dict(native_model.state_dict())
        native_engine, native_optimizer, _, _ = deepspeed.initialize(model=native_model,
                                                                     model_parameters=native_model.parameters(),
                                                                     config=_integration_config(fused=False))
        fused_engine, fused_optimizer, _, _ = deepspeed.initialize(model=fused_model,
                                                                   model_parameters=fused_model.parameters(),
                                                                   config=_integration_config(fused=True))
        assert native_optimizer.fused_gradient_statistics is None
        assert fused_optimizer.fused_gradient_statistics is not None

        torch.manual_seed(1234 + dist.get_rank())
        inputs = torch.randn(1, 8, device=fused_engine.device, dtype=torch.bfloat16)
        targets = torch.randn(1, 8, device=fused_engine.device, dtype=torch.bfloat16)
        fused_engine.backward(fused_engine(inputs, targets))
        native_engine.backward(native_engine(inputs, targets))

        assert not fused_optimizer.has_overflow_partitioned_grads_serial().item()
        assert not native_optimizer.has_overflow_partitioned_grads_serial().item()
        fused_norms = []
        native_norms = []
        for index in range(len(fused_optimizer.bit16_groups)):
            fused_norms.append(
                fused_optimizer.get_grad_norm_direct(fused_optimizer.averaged_gradients[index],
                                                     fused_optimizer.params_in_partition[index]))
            native_norms.append(
                native_optimizer.get_grad_norm_direct(native_optimizer.averaged_gradients[index],
                                                      native_optimizer.params_in_partition[index]))
        torch.testing.assert_close(torch.stack(fused_norms), torch.stack(native_norms))

        global_norm = torch.linalg.vector_norm(torch.stack(fused_norms))
        native_global_norm = torch.linalg.vector_norm(torch.stack(native_norms))
        fused_sample = next(gradient for gradients in fused_optimizer.averaged_gradients.values()
                            for gradient in gradients if gradient is not None).clone()
        native_sample = next(gradient for gradients in native_optimizer.averaged_gradients.values()
                             for gradient in gradients if gradient is not None).clone()
        fused_optimizer.unscale_and_clip_grads([fused_sample], global_norm)
        native_optimizer.unscale_and_clip_grads([native_sample], native_global_norm)
        torch.testing.assert_close(fused_sample, native_sample)

        fused_engine.step()
        native_engine.step()
        assert fused_optimizer.fused_gradient_statistics.refresh_count >= 2
        for fused_parameter, native_parameter in zip(fused_engine.module.parameters(),
                                                     native_engine.module.parameters()):
            torch.testing.assert_close(fused_parameter, native_parameter)
        moment_tensors = [
            state[name] for state in fused_optimizer.optimizer.state.values() for name in ("exp_avg", "exp_avg_sq")
            if name in state
        ]
        assert moment_tensors
        assert all(moment.dtype == torch.bfloat16 for moment in moment_tensors)

        fused_engine.backward(fused_engine(inputs, targets))
        gradient = next(gradient for gradients in fused_optimizer.averaged_gradients.values() for gradient in gradients
                        if gradient is not None)
        if dist.get_rank() == 0:
            gradient.view(-1)[0] = float("nan")
        before_overflow = [parameter.detach().clone() for parameter in fused_engine.module.parameters()]
        skipped_before = fused_engine.skipped_steps
        fused_engine.step()

        assert fused_engine.skipped_steps == skipped_before + 1
        assert all(
            torch.equal(previous, current)
            for previous, current in zip(before_overflow, fused_engine.module.parameters()))
        assert fused_optimizer.fused_gradient_statistics.cache.plan.gradients == ()
        fused_engine.destroy()
        native_engine.destroy()
