# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Compatibility tests for public Triton-op routing and legacy imports."""

import torch
import torch.nn.functional as F


def test_legacy_group_gemm_imports():
    from deepspeed.moe.group_gemm_triton import group_gemm_triton, is_available

    assert callable(group_gemm_triton)
    assert isinstance(is_available(), bool)


def test_swiglu_cpu_falls_back_to_eager():
    from deepspeed.ops.triton_ops import swiglu

    gate = torch.randn(8, 16, dtype=torch.float32, requires_grad=True)
    up = torch.randn(8, 16, dtype=torch.float32, requires_grad=True)
    gate_ref = gate.detach().clone().requires_grad_(True)
    up_ref = up.detach().clone().requires_grad_(True)
    grad_out = torch.randn_like(gate)

    out = swiglu(gate, up)
    out_ref = F.silu(gate_ref) * up_ref
    torch.testing.assert_close(out, out_ref)

    out.backward(grad_out)
    out_ref.backward(grad_out)

    torch.testing.assert_close(gate.grad, gate_ref.grad)
    torch.testing.assert_close(up.grad, up_ref.grad)
