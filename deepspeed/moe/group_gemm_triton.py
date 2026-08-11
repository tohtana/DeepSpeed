# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Compatibility exports for the relocated Triton grouped-GEMM operation."""

from deepspeed.ops.triton_ops._triton import is_triton_available
from deepspeed.ops.triton_ops.group_gemm_triton import group_gemm_triton

__all__ = ["group_gemm_triton", "is_available"]


def is_available() -> bool:
    """Return True if the Triton grouped-GEMM path can be used."""
    return is_triton_available()
