# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

import pytest
import torch

from deepspeed.checkpoint import get_model_3d_descriptor, model_3d_desc
from deepspeed.checkpoint.constants import (CHECKPOINT_PARALLEL_DIMS, CHECKPOINT_PP_DEGREE, CHECKPOINT_TP_DEGREE)


def _write_checkpoint_layout(tmpdir, parallel_dimensions):
    for mp_rank, dimensions in enumerate(parallel_dimensions):
        state = {}
        if dimensions is not None:
            state[CHECKPOINT_PARALLEL_DIMS] = dimensions
        torch.save(state, os.path.join(str(tmpdir), f"mp_rank_{mp_rank:02d}_model_states.pt"))
        for dp_rank in range(2):
            torch.save({},
                       os.path.join(str(tmpdir), f"bf16_zero_pp_rank_{dp_rank}_mp_rank_{mp_rank:02d}_optim_states.pt"))


def _do_reshape(src_3d, tgt_3d):
    assert src_3d.can_reshape(tgt_3d)
    new_3d_map = src_3d.reshape(tgt_3d)

    assert len(new_3d_map) == tgt_3d.dp_degree
    for new_2d_map in new_3d_map:
        assert new_2d_map.pp_degree == tgt_3d.pp_degree
        assert new_2d_map.tp_degree == tgt_3d.tp_degree

    return new_3d_map


# Specify 3d shape as pp/tp/dp
def test_reshape_222_to_111():
    src_3d = model_3d_desc(pp_degree=2, tp_degree=2, dp_degree=2)
    tgt_3d = model_3d_desc(pp_degree=1, tp_degree=1, dp_degree=1)

    new_3d_map = _do_reshape(src_3d, tgt_3d)

    assert new_3d_map[0].get_data(pp_index=0, tp_index=0) == [0, 4, 1, 5, 2, 6, 3, 7]


def test_reshape_222_to_121():
    src_3d = model_3d_desc(pp_degree=2, tp_degree=2, dp_degree=2)
    tgt_3d = model_3d_desc(pp_degree=1, tp_degree=2, dp_degree=1)

    new_3d_map = _do_reshape(src_3d, tgt_3d)

    assert new_3d_map[0].get_data(pp_index=0, tp_index=0) == [0, 4, 2, 6]
    assert new_3d_map[0].get_data(pp_index=0, tp_index=1) == [1, 5, 3, 7]


def test_reshape_222_to_122():
    src_3d = model_3d_desc(pp_degree=2, tp_degree=2, dp_degree=2)
    tgt_3d = model_3d_desc(pp_degree=1, tp_degree=2, dp_degree=2)

    new_3d_map = _do_reshape(src_3d, tgt_3d)

    assert new_3d_map[0].get_data(pp_index=0, tp_index=0) == [0, 4]
    assert new_3d_map[0].get_data(pp_index=0, tp_index=1) == [1, 5]
    assert new_3d_map[1].get_data(pp_index=0, tp_index=0) == [2, 6]
    assert new_3d_map[1].get_data(pp_index=0, tp_index=1) == [3, 7]


def test_reshape_222_to_211():
    src_3d = model_3d_desc(pp_degree=2, tp_degree=2, dp_degree=2)
    tgt_3d = model_3d_desc(pp_degree=2, tp_degree=1, dp_degree=1)

    new_3d_map = _do_reshape(src_3d, tgt_3d)

    assert new_3d_map[0].get_data(pp_index=0, tp_index=0) == [0, 4, 1, 5]
    assert new_3d_map[0].get_data(pp_index=1, tp_index=0) == [2, 6, 3, 7]


def test_checkpoint_descriptor_uses_explicit_parallel_dimensions(tmpdir):
    dimensions = {
        CHECKPOINT_PP_DEGREE: 2,
        CHECKPOINT_TP_DEGREE: 1,
    }
    _write_checkpoint_layout(tmpdir, [dimensions, dimensions])

    descriptor = get_model_3d_descriptor(str(tmpdir))

    assert descriptor.pp_degree == 2
    assert descriptor.tp_degree == 1
    assert descriptor.dp_degree == 2


def test_checkpoint_descriptor_preserves_legacy_inference(tmpdir):
    _write_checkpoint_layout(tmpdir, [None, None])

    descriptor = get_model_3d_descriptor(str(tmpdir))

    assert descriptor.pp_degree == 1
    assert descriptor.tp_degree == 2
    assert descriptor.dp_degree == 2


def test_checkpoint_descriptor_rejects_partial_parallel_dimensions(tmpdir):
    dimensions = {
        CHECKPOINT_PP_DEGREE: 2,
        CHECKPOINT_TP_DEGREE: 1,
    }
    _write_checkpoint_layout(tmpdir, [dimensions, None])

    with pytest.raises(RuntimeError, match="missing from model-state files"):
        get_model_3d_descriptor(str(tmpdir))
