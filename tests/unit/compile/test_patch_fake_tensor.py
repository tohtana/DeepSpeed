# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.compile.patch_fake_tensor import _get_guard_sizes_strides, set_z3_module_hooks_kept


def test_zero3_param_guards_use_released_shape_without_module_hooks():
    param = torch.nn.Parameter(torch.empty(4, 3))
    param.ds_id = 1

    try:
        set_z3_module_hooks_kept(False)

        size, stride = _get_guard_sizes_strides(param)

        assert size == torch.Size([0])
        assert stride == (1, )
    finally:
        set_z3_module_hooks_kept(False)


def test_zero3_param_guards_use_current_shape_with_module_hooks():
    param = torch.nn.Parameter(torch.empty(4, 3))
    param.ds_id = 1

    try:
        set_z3_module_hooks_kept(True)

        size, stride = _get_guard_sizes_strides(param)

        assert size == torch.Size([4, 3])
        assert stride == (3, 1)
    finally:
        set_z3_module_hooks_kept(False)
