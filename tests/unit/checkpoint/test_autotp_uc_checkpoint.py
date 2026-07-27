# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import types
from types import SimpleNamespace

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.checkpoint.ds_to_universal import main as convert_to_universal
from deepspeed.checkpoint.constants import (CAT_DIM, FP32_WEIGHT_KEY, PARAM, PARAM_SHAPES,
                                            PARAMETER_WITH_ROW_PARALLELISM_PATTERNS, PARAMETER_WITH_SUB_PARAMS,
                                            SUB_PARAM_SHAPE, TP_REPLICATED_PARAMETER_PATTERNS,
                                            UNIVERSAL_CHECKPOINT_INFO)
from deepspeed.checkpoint.universal_checkpoint import SubparamShape as CheckpointSubparamShape
from deepspeed.checkpoint.ds_to_universal import _collect_slice_shapes, merge_tp_slices
from deepspeed.checkpoint.universal_checkpoint import (_get_param_uc_restore_meta, _resolve_autotp_partition,
                                                       load_hp_checkpoint_state)
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

from unit.common import DistributedTest


class _DummyAddress:

    def __init__(self, start, numel):
        self.start = start
        self.numel = numel


class _DummyHPMapping:

    def __init__(self, param):
        self.lp_fragment_address = _DummyAddress(0, param.numel())
        self._param = param
        self.optim_fragment = {}

    def get_hp_fragment(self):
        return self._param.view(-1)

    def get_optim_state_keys(self):
        return []


def _make_param(shape, meta=None):
    param = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float32))
    param._hp_mapping = _DummyHPMapping(param)
    if meta is not None:
        setattr(param, 'ds_autotp_universal_checkpoint_meta', meta)
    return param


def test_resolve_autotp_partition_row_parallel_weight():
    param = _make_param(
        (4, 4), {
            'partition_type': 'row',
            'partition_dim': 1,
            'logical_shape': (4, 8),
            'output_shape': (4, ),
            'sub_param_shape': None,
            'original_shape': (4, 8),
            'is_bias': False,
            'replicated': False,
        })
    full_hp_param = torch.arange(32, dtype=torch.float32).view(4, 8)

    slice_flat = _resolve_autotp_partition(param, {PARAM: full_hp_param}, full_hp_param, tp_rank=1, tp_world_size=2)

    expected = full_hp_param.chunk(2, dim=1)[1].flatten()
    assert torch.equal(slice_flat, expected)


def test_resolve_autotp_partition_subparam_column_weight():
    param = _make_param(
        (3, 4), {
            'partition_type': 'column',
            'partition_dim': 0,
            'logical_shape': (6, 4),
            'output_shape': (6, ),
            'sub_param_shape': ((2, 2, 2), 4),
            'original_shape': (6, 4),
            'is_bias': False,
            'replicated': False,
        })
    full_hp_param = torch.arange(24, dtype=torch.float32).view(6, 4)

    slice_flat = _resolve_autotp_partition(param, {PARAM: full_hp_param}, full_hp_param, tp_rank=0, tp_world_size=2)

    chunks = [sub.chunk(2, dim=0)[0] for sub in full_hp_param.view(3, 2, 4)]
    expected = torch.cat(chunks, dim=0).flatten()
    assert torch.equal(slice_flat, expected)


def test_resolve_autotp_partition_subparam_sizes_uneven_gqa_like():
    # Simulate a fused QKV weight where Q/K/V have uneven sizes along partition_dim=0.
    # Example (GQA-like):
    #   Q: 8
    #   K: 4
    #   V: 4
    # Total: 16
    #
    # With tp_world_size=2, correct slicing is:
    #   Q chunk -> 4 per rank
    #   K chunk -> 2 per rank
    #   V chunk -> 2 per rank
    # Each rank gets 8 rows total, but importantly boundaries must align with Q/K/V.
    sub_param_sizes = [8, 4, 4]
    tp_world_size = 2
    tp_rank = 1

    param = _make_param(
        (8, 2),
        {
            "partition_type": "column",
            "partition_dim": 0,
            "logical_shape": (sum(sub_param_sizes), 2),  # (16, 2)
            "output_shape": (sum(sub_param_sizes), ),  # (16,)
            "sub_param_shape": (tuple(sub_param_sizes), 2),
            "sub_param_sizes": sub_param_sizes,
            "original_shape": (sum(sub_param_sizes), 2),
            "is_bias": False,
            "replicated": False,
        })

    # Full (unsharded) HP parameter: shape (16, 2)
    full_hp_param = torch.arange(sum(sub_param_sizes) * 2, dtype=torch.float32).view(sum(sub_param_sizes), 2)

    slice_flat = _resolve_autotp_partition(param, {PARAM: full_hp_param},
                                           full_hp_param,
                                           tp_rank=tp_rank,
                                           tp_world_size=tp_world_size)

    # Expected: split into Q/K/V blocks, chunk each block by TP, take tp_rank slice, concat back.
    q, k, v = torch.split(full_hp_param, sub_param_sizes, dim=0)
    expected = torch.cat([
        q.chunk(tp_world_size, dim=0)[tp_rank],
        k.chunk(tp_world_size, dim=0)[tp_rank],
        v.chunk(tp_world_size, dim=0)[tp_rank]
    ],
                         dim=0).flatten()

    assert torch.equal(slice_flat, expected)


def test_resolve_autotp_partition_uses_uneven_partition_sizes():
    full_hp_param = torch.arange(101 * 4, dtype=torch.float32).view(101, 4)
    param = _make_param(
        (50, 4), {
            'partition_type': 'column',
            'partition_dim': 0,
            'logical_shape': (101, 4),
            'output_shape': (101, ),
            'partition_sizes': (51, 50),
            'original_shape': (101, 4),
            'is_bias': False,
            'replicated': False,
        })

    slice_flat = _resolve_autotp_partition(param, {PARAM: full_hp_param}, full_hp_param, tp_rank=1, tp_world_size=2)

    expected = full_hp_param.narrow(0, 51, 50).flatten()
    assert torch.equal(slice_flat, expected)


def test_resolve_autotp_partition_uses_uneven_partition_sizes_for_bias():
    full_hp_param = torch.arange(101, dtype=torch.float32)
    param = _make_param(
        (50, ), {
            'partition_type': 'column',
            'partition_dim': 0,
            'logical_shape': (101, ),
            'output_shape': (101, ),
            'partition_sizes': (51, 50),
            'original_shape': (101, ),
            'is_bias': True,
            'replicated': False,
        })

    slice_flat = _resolve_autotp_partition(param, {PARAM: full_hp_param}, full_hp_param, tp_rank=1, tp_world_size=2)

    expected = full_hp_param.narrow(0, 51, 50).flatten()
    assert torch.equal(slice_flat, expected)


def test_resolve_autotp_partition_replicated_bias():
    full_hp_param = torch.arange(8, dtype=torch.float32)
    param = _make_param(
        (8, ), {
            'partition_type': 'row',
            'partition_dim': None,
            'logical_shape': (8, ),
            'output_shape': (8, ),
            'sub_param_shape': None,
            'original_shape': (8, ),
            'is_bias': True,
            'replicated': True,
        })

    slice_flat = _resolve_autotp_partition(param, {PARAM: full_hp_param}, full_hp_param, tp_rank=1, tp_world_size=2)

    assert torch.equal(slice_flat, full_hp_param)


def test_load_hp_checkpoint_state_prefers_autotp_metadata(tmp_path, monkeypatch):
    param = _make_param(
        (4, 4), {
            'partition_type': 'row',
            'partition_dim': 1,
            'logical_shape': (4, 8),
            'output_shape': (4, ),
            'sub_param_shape': None,
            'original_shape': (4, 8),
            'is_bias': False,
            'replicated': False,
        })
    param.load_hp_checkpoint_state = types.MethodType(load_hp_checkpoint_state, param)

    import deepspeed.checkpoint.universal_checkpoint as uc
    monkeypatch.setattr(uc, "current_param", param, raising=False)

    ckpt_dir = tmp_path / "weight"
    ckpt_dir.mkdir(parents=True)
    full_hp_param = torch.arange(32, dtype=torch.float32).view(4, 8)
    torch.save({PARAM: full_hp_param}, ckpt_dir / f"{FP32_WEIGHT_KEY}.pt")

    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {PARAM: full_hp_param} if str(args[0]).endswith("fp32.pt") else 0,
    )

    step = param.load_hp_checkpoint_state(str(ckpt_dir), tp_rank=1, tp_world_size=2)

    assert step is None
    expected = full_hp_param.chunk(2, dim=1)[1].flatten()
    assert torch.equal(param.data.flatten(), expected)


def _write_tp_slice(base_dir, param_name, tp_idx, state_name, tensor):
    shard_dir = base_dir / param_name / str(tp_idx)
    shard_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.reshape(-1), shard_dir / f"{state_name}.00")


def _write_tp_states(base_dir, param_name, tp_idx, fp32_tensor):
    # merge_tp_slices attempts to merge these three states, so the test must write all of them.
    _write_tp_slice(base_dir, param_name, tp_idx, "fp32", fp32_tensor)
    _write_tp_slice(base_dir, param_name, tp_idx, "exp_avg", torch.zeros_like(fp32_tensor))
    _write_tp_slice(base_dir, param_name, tp_idx, "exp_avg_sq", torch.zeros_like(fp32_tensor))


def test_merge_tp_slices_emits_subparam_shape_metadata(tmp_path):
    slice_dir = tmp_path / "slices"
    output_dir = tmp_path / "out"
    param_name = "module.qkv.weight"

    tp0 = torch.arange(12, dtype=torch.float32).view(3, 4)
    tp1 = torch.arange(12, 24, dtype=torch.float32).view(3, 4)
    _write_tp_states(slice_dir, param_name, 0, tp0)
    _write_tp_states(slice_dir, param_name, 1, tp1)

    uc_info = {
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [],
        TP_REPLICATED_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [{
            "patterns": [rf"^{param_name}$"],
            "shape": [(2, 2, 2), 4],
            "partition_dim": 0,
        }],
    }

    ds_checkpoint = SimpleNamespace(
        get_checkpoint_info=lambda key: uc_info if key == UNIVERSAL_CHECKPOINT_INFO else {})

    unmatched = merge_tp_slices(ds_checkpoint, str(output_dir), str(slice_dir), 2,
                                (param_name, [torch.Size([3, 4]), torch.Size([3, 4])]))

    ckpt = torch.load(output_dir / param_name / "fp32.pt", weights_only=False)
    assert not unmatched
    assert isinstance(ckpt[SUB_PARAM_SHAPE], CheckpointSubparamShape)
    assert ckpt[SUB_PARAM_SHAPE].partition_dim == 0


def test_merge_tp_slices_uses_row_parallel_cat_dim(tmp_path):
    slice_dir = tmp_path / "slices"
    output_dir = tmp_path / "out"
    param_name = "module.proj.weight"

    # Uneven row-parallel shards: rank 0 owns 3 input columns, rank 1 owns 2.
    tp0 = torch.arange(12, dtype=torch.float32).view(4, 3)
    tp1 = torch.arange(12, 20, dtype=torch.float32).view(4, 2)
    _write_tp_states(slice_dir, param_name, 0, tp0)
    _write_tp_states(slice_dir, param_name, 1, tp1)

    uc_info = {
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [rf"^{param_name}$"],
        TP_REPLICATED_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [],
    }

    ds_checkpoint = SimpleNamespace(
        get_checkpoint_info=lambda key: uc_info if key == UNIVERSAL_CHECKPOINT_INFO else {})

    merge_tp_slices(ds_checkpoint, str(output_dir), str(slice_dir), 2,
                    (param_name, [torch.Size([4, 3]), torch.Size([4, 2])]))

    ckpt = torch.load(output_dir / param_name / "fp32.pt", weights_only=False)
    assert ckpt[CAT_DIM] == 1
    assert torch.equal(ckpt[PARAM], torch.cat([tp0, tp1], dim=1))


def test_zero_optimizer_uc_info_comes_from_cached_state():
    param = _make_param((2, 2))
    expected_uc_info = {"key": "value"}
    setattr(param, UNIVERSAL_CHECKPOINT_INFO, expected_uc_info)

    optimizer = object.__new__(DeepSpeedZeroOptimizer)
    optimizer.bit16_groups = [[param]]
    optimizer._enable_universal_checkpoint()
    delattr(param, UNIVERSAL_CHECKPOINT_INFO)

    assert optimizer._get_universal_checkpoint_info() == expected_uc_info


def test_bf16_optimizer_uc_info_comes_from_cached_state():
    param = _make_param((2, 2))
    expected_uc_info = {"key": "value"}
    setattr(param, UNIVERSAL_CHECKPOINT_INFO, expected_uc_info)

    optimizer = object.__new__(BF16_Optimizer)
    optimizer.bf16_groups = [[param]]
    optimizer._enable_universal_checkpoint()
    delattr(param, UNIVERSAL_CHECKPOINT_INFO)

    assert optimizer._get_universal_checkpoint_info() == expected_uc_info


def test_get_param_uc_restore_meta_returns_top_level_restore_schema():
    meta = {
        "partition_dim": 1,
        "logical_shape": (4, 8),
        "output_shape": (4, ),
        "sub_param_shape": None,
        "sub_param_sizes": None,
        "target_partition_shape": (4, 4),
        "is_bias": False,
        "replicated": False,
        "conversion": {
            "partition_dim": 999
        },
    }
    param = _make_param((4, 4), meta)

    restore_meta = _get_param_uc_restore_meta(param)

    assert restore_meta["partition_dim"] == 1
    assert restore_meta["conversion"]["partition_dim"] == 999


CP_TAG = "uneven_tp"
UNIVERSAL_TAG = f"{CP_TAG}_universal"


class UnevenVocabLmHeadModel(torch.nn.Module):

    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        return self.lm_head(x).sum()


class GQAAttentionModel(torch.nn.Module):
    """Column-parallel q/k/v feeding a row-parallel o_proj, sharded on kv-head boundaries."""

    class Config:

        def __init__(self, hidden_dim, num_heads):
            self.hidden_size = hidden_dim
            self.num_attention_heads = num_heads
            self.num_key_value_heads = num_heads

    class Attention(torch.nn.Module):

        def __init__(self, hidden_dim):
            super().__init__()
            self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

        def forward(self, x):
            return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))

    class Layer(torch.nn.Module):

        def __init__(self, hidden_dim):
            super().__init__()
            self.self_attn = GQAAttentionModel.Attention(hidden_dim)

        def forward(self, x):
            return self.self_attn(x)

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.layers = torch.nn.ModuleList([GQAAttentionModel.Layer(hidden_dim)])
        self.config = GQAAttentionModel.Config(hidden_dim, num_heads)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.sum()


def _convert_to_universal(checkpoint_dir, universal_dir):
    convert_to_universal(
        SimpleNamespace(input_folder=checkpoint_dir,
                        output_folder=universal_dir,
                        num_extract_workers=1,
                        num_merge_workers=1,
                        keep_temp_folder=False,
                        strict=True,
                        inject_missing_state=False))


def _train_steps(engine, hidden_dim, steps=3):
    for _ in range(steps):
        batch = torch.randn(2, hidden_dim, device=engine.device)
        dist.broadcast(batch, src=0)
        engine.backward(engine(batch))
        engine.step()


def _save_and_convert(engine, tmpdir):
    engine.save_checkpoint(tmpdir, tag=CP_TAG, client_state={"iteration": 3})
    dist.barrier()
    if dist.get_rank() == 0:
        _convert_to_universal(os.path.join(tmpdir, CP_TAG), os.path.join(tmpdir, UNIVERSAL_TAG))
    dist.barrier()


class TestUnevenColumnUniversalCheckpoint(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def test_save_convert_load_uneven_lm_head(self, tmpdir):
        hidden_dim = 12
        vocab_size = 101  # Not divisible by the two TP ranks, giving shards of 51 and 50.
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "tensor_parallel": {
                "autotp_size": self.world_size,
                "partition_config": {
                    "use_default_specs":
                    False,
                    "layer_specs": [{
                        "patterns": [r".*lm_head\.weight$"],
                        "partition_type": "column",
                        "gather_output": True,
                    }],
                },
            },
            "zero_optimization": {
                "stage": 1
            },
        }

        torch.manual_seed(42)
        model = UnevenVocabLmHeadModel(hidden_dim, vocab_size)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        assert engine.module.lm_head.weight.shape[0] in (51, 50)

        _train_steps(engine, hidden_dim)
        expected_weight = engine.module.lm_head.weight.detach().cpu().clone()
        expected_bias = engine.module.lm_head.bias.detach().cpu().clone()
        _save_and_convert(engine, tmpdir)

        if dist.get_rank() == 0:
            merged = torch.load(os.path.join(tmpdir, UNIVERSAL_TAG, "zero", "lm_head.weight", "fp32.pt"),
                                weights_only=False)
            assert tuple(merged[PARAM].shape) == (vocab_size, hidden_dim)

        config_dict["checkpoint"] = {"load_universal": True}
        torch.manual_seed(123)
        restored = UnevenVocabLmHeadModel(hidden_dim, vocab_size)
        restored_engine, _, _, _ = deepspeed.initialize(model=restored,
                                                        model_parameters=restored.parameters(),
                                                        config=config_dict)
        restored_engine.load_checkpoint(tmpdir, tag=UNIVERSAL_TAG, load_optimizer_states=True)

        torch.testing.assert_close(restored_engine.module.lm_head.weight.detach().cpu(), expected_weight)
        torch.testing.assert_close(restored_engine.module.lm_head.bias.detach().cpu(), expected_bias)

        # The optimizer must be usable after the restore.
        _train_steps(restored_engine, hidden_dim, steps=1)


class TestUnevenRowUniversalCheckpoint(DistributedTest):
    world_size = 4
    reuse_dist_env = False

    def test_save_convert_load_uneven_row_parallel(self, tmpdir):
        hidden_dim = 384
        num_heads = 6  # Not divisible by the four TP ranks, giving shards of 128/128/64/64.
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "tensor_parallel": {
                "autotp_size": self.world_size
            },
            "zero_optimization": {
                "stage": 1
            },
        }

        torch.manual_seed(42)
        model = GQAAttentionModel(hidden_dim, num_heads)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        attn = engine.module.layers[0].self_attn
        # Column and row parallelism must shard the same dimension identically.
        assert attn.q_proj.weight.shape[0] == attn.o_proj.weight.shape[1]

        _train_steps(engine, hidden_dim)
        expected_q = attn.q_proj.weight.detach().cpu().clone()
        expected_o = attn.o_proj.weight.detach().cpu().clone()
        _save_and_convert(engine, tmpdir)

        if dist.get_rank() == 0:
            merged = torch.load(os.path.join(tmpdir, UNIVERSAL_TAG, "zero", "layers.0.self_attn.o_proj.weight",
                                             "fp32.pt"),
                                weights_only=False)
            assert tuple(merged[PARAM].shape) == (hidden_dim, hidden_dim)

        config_dict["checkpoint"] = {"load_universal": True}
        torch.manual_seed(123)
        restored = GQAAttentionModel(hidden_dim, num_heads)
        restored_engine, _, _, _ = deepspeed.initialize(model=restored,
                                                        model_parameters=restored.parameters(),
                                                        config=config_dict)
        restored_engine.load_checkpoint(tmpdir, tag=UNIVERSAL_TAG, load_optimizer_states=True)

        restored_attn = restored_engine.module.layers[0].self_attn
        torch.testing.assert_close(restored_attn.q_proj.weight.detach().cpu(), expected_q)
        torch.testing.assert_close(restored_attn.o_proj.weight.detach().cpu(), expected_o)

        _train_steps(restored_engine, hidden_dim, steps=1)


def _write_mp_rank_file(dir_path, mp_rank, param_shapes):
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"mp_rank_{mp_rank:02d}_model_states.pt")
    torch.save({PARAM_SHAPES: param_shapes}, path)
    return path


def test_collect_slice_shapes_keeps_uneven_shapes_in_tp_order(tmp_path):
    # tp=2 with a column dimension of 5, giving shards of 3 and 2.
    files = [
        _write_mp_rank_file(tmp_path, 0, [{
            "lm_head.weight": torch.Size([3, 4])
        }]),
        _write_mp_rank_file(tmp_path, 1, [{
            "lm_head.weight": torch.Size([2, 4])
        }]),
    ]
    ds_checkpoint = SimpleNamespace(mp_rank_files=files, tp_degree=2, pp_degree=1)

    shapes = _collect_slice_shapes(ds_checkpoint)

    assert shapes["lm_head.weight"] == [torch.Size([3, 4]), torch.Size([2, 4])]


def test_collect_slice_shapes_pipeline_parallel_layout(tmp_path):
    # Model-parallel ranks enumerate the (pp, tp) grid with tp varying fastest, and each
    # pipeline stage only owns its own parameters.
    stage0 = {"layers.0.weight": torch.Size([3, 4])}, {"layers.0.weight": torch.Size([2, 4])}
    stage1 = {"layers.1.weight": torch.Size([3, 4])}, {"layers.1.weight": torch.Size([2, 4])}
    files = [
        _write_mp_rank_file(tmp_path, 0, [stage0[0]]),
        _write_mp_rank_file(tmp_path, 1, [stage0[1]]),
        _write_mp_rank_file(tmp_path, 2, [stage1[0]]),
        _write_mp_rank_file(tmp_path, 3, [stage1[1]]),
    ]
    ds_checkpoint = SimpleNamespace(mp_rank_files=files, tp_degree=2, pp_degree=2)

    shapes = _collect_slice_shapes(ds_checkpoint)

    # Every parameter is collected once per tp rank, in tp order, despite spanning two stages.
    assert shapes["layers.0.weight"] == [torch.Size([3, 4]), torch.Size([2, 4])]
    assert shapes["layers.1.weight"] == [torch.Size([3, 4]), torch.Size([2, 4])]


def test_collect_slice_shapes_rejects_unexpected_rank_count(tmp_path):
    files = [_write_mp_rank_file(tmp_path, 0, [{"lm_head.weight": torch.Size([3, 4])}])]
    ds_checkpoint = SimpleNamespace(mp_rank_files=files, tp_degree=2, pp_degree=1)

    with pytest.raises(AssertionError, match="one per tp rank"):
        _collect_slice_shapes(ds_checkpoint)
