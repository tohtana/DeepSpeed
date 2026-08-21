# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.module_inject import tp_shard
from deepspeed.module_inject.tp_shard import AutoTPMeta, get_shard_size, get_shard_size_list


@pytest.mark.parametrize("total_size,tp_size", [(50257, 2), (50257, 8), (151936, 8), (32000, 4)])
def test_grain_quantized_shards_tile_the_dimension(total_size, tp_size):
    # A vocabulary that is not a multiple of tp_grain_size used to lose its tail to the grain
    # quantization, so the shards no longer reconstructed the embedding table.
    meta = AutoTPMeta(tp_grain_size=64)

    shard_sizes = get_shard_size_list(total_size, tp_size, meta, "lm_head")

    assert sum(shard_sizes) == total_size
    # Only the rank that absorbs the sub-grain tail gives up its alignment.
    assert sum(1 for size in shard_sizes if size % 64) <= 1, shard_sizes


def test_uneven_shards_without_grain_quantization():
    assert get_shard_size_list(101, 2, AutoTPMeta(), "lm_head") == [51, 50]


def test_kv_head_shards_tile_the_dimension():
    meta = AutoTPMeta(num_kv_heads=6)

    # 6 kv heads over 4 ranks gives 2/2/1/1 heads, so 384 hidden splits as 128/128/64/64.
    assert get_shard_size_list(384, 4, meta, "layers.0.self_attn.q_proj") == [128, 128, 64, 64]


def test_two_models_do_not_clobber_each_others_meta():
    # Each model carries its own AutoTPMeta, so loading a second model does not re-shard the
    # first one.
    model_a = AutoTPMeta(num_kv_heads=6, tp_grain_size=64)
    model_b = AutoTPMeta(num_kv_heads=2, tp_grain_size=1)

    a_qproj = get_shard_size_list(384, 4, model_a, "layers.0.self_attn.q_proj")
    a_lmhead = get_shard_size_list(1001, 2, model_a, "lm_head")

    # A second model is loaded into the same process.
    _ = get_shard_size_list(384, 4, model_b, "layers.0.self_attn.q_proj")

    # Model A's partition contract is unchanged.
    assert get_shard_size_list(384, 4, model_a, "layers.0.self_attn.q_proj") == a_qproj
    assert get_shard_size_list(1001, 2, model_a, "lm_head") == a_lmhead


def test_process_group_resolves_noncontiguous_group_rank(monkeypatch):
    meta = AutoTPMeta(tp_grain_size=64)
    tp_group = object()
    monkeypatch.setattr(tp_shard.dist, "get_rank", lambda group=None: 1 if group is tp_group else 2)

    shard_sizes = get_shard_size_list(50257, 2, meta, "lm_head")
    assert get_shard_size(50257, 2, meta, "lm_head", mp_group=tp_group) == shard_sizes[1]


def test_shard_size_refuses_to_guess_subgroup_rank(monkeypatch):
    monkeypatch.setattr(tp_shard.dist, "get_world_size", lambda: 4)

    with pytest.raises(ValueError, match="group-local rank or process group"):
        get_shard_size(12, 2, AutoTPMeta())
