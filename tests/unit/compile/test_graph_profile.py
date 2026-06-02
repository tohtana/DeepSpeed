# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

from deepspeed.compile.profilers.graph_profile import _should_partition_profile_param


def test_profile_partitions_inactive_zero3_param():
    param = SimpleNamespace(ds_id=1, ds_persist=False, ds_active_sub_modules=set())

    assert _should_partition_profile_param(param)


def test_profile_does_not_partition_persistent_zero3_param():
    param = SimpleNamespace(ds_id=1, ds_persist=True, ds_active_sub_modules=set())

    assert not _should_partition_profile_param(param)


def test_profile_does_not_partition_module_hook_active_zero3_param():
    param = SimpleNamespace(ds_id=1, ds_persist=False, ds_active_sub_modules={42})

    assert not _should_partition_profile_param(param)
