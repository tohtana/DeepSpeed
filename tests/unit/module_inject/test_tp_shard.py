# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.module_inject import tp_shard
from deepspeed.module_inject.tp_shard import get_shard_size_list, set_num_kv_heads, set_tp_grain_size


@pytest.fixture(autouse=True)
def restore_tp_shard_globals():
    # tp_grain_size and num_kv_heads are process wide, so leaking them would change how
    # unrelated tests partition their layers.
    grain_size, kv_heads = tp_shard.tp_grain_size, tp_shard.num_kv_heads
    yield
    set_tp_grain_size(grain_size)
    set_num_kv_heads(kv_heads)


@pytest.mark.parametrize("total_size,tp_size", [(50257, 2), (50257, 8), (151936, 8), (32000, 4)])
def test_grain_quantized_shards_tile_the_dimension(total_size, tp_size):
    # A vocabulary that is not a multiple of tp_grain_size used to lose its tail to the grain
    # quantization, so the shards no longer reconstructed the embedding table.
    set_tp_grain_size(64)

    shard_sizes = get_shard_size_list(total_size, tp_size, "lm_head")

    assert sum(shard_sizes) == total_size
    # Only the rank that absorbs the sub-grain tail gives up its alignment.
    assert sum(1 for size in shard_sizes if size % 64) <= 1, shard_sizes


def test_uneven_shards_without_grain_quantization():
    assert get_shard_size_list(101, 2, "lm_head") == [51, 50]


def test_kv_head_shards_tile_the_dimension():
    set_num_kv_heads(6)

    # 6 kv heads over 4 ranks gives 2/2/1/1 heads, so 384 hidden splits as 128/128/64/64.
    assert get_shard_size_list(384, 4, "layers.0.self_attn.q_proj") == [128, 128, 64, 64]
