# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass
from typing import Optional

from deepspeed import comm as dist


@dataclass(frozen=True)
class AutoTPMeta:
    """Per-model tensor-parallel metadata AutoTP derives from the model config.

    Each model carries its own kv-head / grain values so its layers, fused-QKV repacking and
    checkpoint conversion shard consistently, and more than one AutoTP model can live in the
    same process.
    """
    num_kv_heads: Optional[int] = None
    num_attention_heads: Optional[int] = None
    n_embd: Optional[int] = None
    tp_grain_size: int = 1

    @classmethod
    def from_model_config(cls, model_config, tp_grain_size: int = 1) -> "AutoTPMeta":
        """The single source of truth for reading kv-head / hidden / attention-head counts.

        ``model_config`` may be ``None`` for callers that build a synthetic AutoTP with no real
        model (some unit tests); they get an empty meta and the resulting even-grain split.
        """
        if model_config is None:
            return cls(tp_grain_size=tp_grain_size)
        # multi_query_group_num is for chatglm2 & chatglm3
        num_kv_heads = None
        for name in ('multi_query_group_num', 'num_kv_heads', 'num_key_value_heads', 'num_attention_heads', 'n_heads',
                     'attention_heads'):
            if hasattr(model_config, name):
                num_kv_heads = getattr(model_config, name)
                if num_kv_heads is not None:
                    break
        n_embd = None
        for name in ('n_embd', 'hidden_size'):
            if hasattr(model_config, name):
                n_embd = getattr(model_config, name)
            if n_embd is not None:
                break
        num_attention_heads = getattr(model_config, 'num_attention_heads', None)
        return cls(num_kv_heads=num_kv_heads,
                   num_attention_heads=num_attention_heads,
                   n_embd=n_embd,
                   tp_grain_size=tp_grain_size)


def get_shard_size(total_size, mp_size, meta: AutoTPMeta, name=None, rank=None, mp_group=None, num_kv_heads=None):
    """Size of one shard of ``total_size`` split across a tensor-parallel group of ``mp_size``.

    ``meta`` carries this model's ``num_kv_heads`` / ``tp_grain_size`` so the split is stable
    for the lifetime of the model instead of depending on whichever AutoTP model was loaded
    last.

    ``rank`` is the rank *within the tensor-parallel group*, i.e. in ``[0, mp_size)``, matching
    ``dist.get_rank(group=mp_group)`` and the index used by ``get_shard_size_list``. It is not a
    global rank.

    ``num_kv_heads`` overrides ``meta.num_kv_heads`` when set. Passing it explicitly lets callers
    split fused sub-parameters (Q/K/V) against their respective head counts without
    reimplementing the KV-head-aligned partition logic at the call site.
    """
    if num_kv_heads is None:
        num_kv_heads = meta.num_kv_heads
    tp_grain_size = meta.tp_grain_size
    last_linear = ["lm_head", "embed_out"]
    # MoE MLP layer use near even division will get better perf.
    moe_mlp_layer = ["gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"]
    not_moe_mlp_layer = True
    if name != None and any(s in str(name) for s in moe_mlp_layer):
        not_moe_mlp_layer = False
    # When num_kv_heads is defined, uneven division is possible, otherwise enforce near even division
    if rank is None:
        if mp_group is not None:
            rank = dist.get_rank(group=mp_group)
        else:
            world_size = dist.get_world_size()
            if world_size != mp_size:
                raise ValueError("get_shard_size requires a group-local rank or process group when mp_size "
                                 f"({mp_size}) differs from the distributed world size ({world_size}).")
            rank = dist.get_rank()
    if num_kv_heads is not None and total_size % num_kv_heads == 0 and "mlp" not in str(name) and \
        str(name) not in last_linear and not_moe_mlp_layer:
        my_slices = (num_kv_heads // mp_size) + (1 if rank < (num_kv_heads % mp_size) else 0)
        return total_size * my_slices // num_kv_heads
    else:
        if total_size >= tp_grain_size:
            grain_size, remainder = divmod(total_size, tp_grain_size)
            shard_size = (grain_size // mp_size + (1 if rank < (grain_size % mp_size) else 0)) * tp_grain_size
            if rank == mp_size - 1:
                # Quantizing to tp_grain_size would otherwise drop total_size % tp_grain_size
                # and silently truncate the dimension. Giving that tail to the last rank keeps
                # every other rank aligned for the compute kernels.
                shard_size += remainder
            return shard_size
        else:
            return total_size // mp_size + (1 if rank < (total_size % mp_size) else 0)


def get_shard_size_list(total_size, mp_size, meta: AutoTPMeta, name=None, num_kv_heads=None):
    shard_sizes = []
    if num_kv_heads is None:
        num_kv_heads = meta.num_kv_heads
    for i in range(mp_size):
        shard_sizes.append(get_shard_size(total_size, mp_size, meta, name, i, num_kv_heads=num_kv_heads))
    # Shards must tile the dimension exactly, otherwise the partitioned weights no longer
    # reconstruct the original tensor.
    assert sum(shard_sizes) == total_size, (
        f"AutoTP shard sizes {shard_sizes} for layer '{name}' do not sum to the dimension size "
        f"{total_size} with tp_size={mp_size}, tp_grain_size={meta.tp_grain_size} and "
        f"num_kv_heads={num_kv_heads}.")
    return shard_sizes
