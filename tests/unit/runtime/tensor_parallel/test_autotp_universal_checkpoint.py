# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.checkpoint.constants import (PARAMETER_WITH_ROW_PARALLELISM_PATTERNS, PARAMETER_WITH_SUB_PARAMS,
                                            SUB_PARAM_SHARD_WIDTHS, TP_REPLICATED_PARAMETER_PATTERNS,
                                            VOCABULARY_PARAMETER_PATTERNS, DS_AUTOTP_UC_META)
from deepspeed.module_inject.layers import (_build_param_uc_restore_meta, _get_param_uc_conversion_meta,
                                            GateUpPack_LinearLayer, LinearAllreduce, LinearLayer, SubParamLinearLayer,
                                            fused_LinearLayer, collect_autotp_universal_checkpoint_info)


def test_collect_autotp_universal_checkpoint_info_row_parallel():
    layer = LinearAllreduce(torch.nn.Linear(16, 8, bias=True), mp_group=None, name="proj")
    model = torch.nn.Module()
    model.proj = layer

    uc_info = collect_autotp_universal_checkpoint_info(model)

    # collect_autotp_universal_checkpoint_info() stores regex patterns like r"^proj\.weight$"
    assert r"^proj\.weight$" in uc_info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS]
    # bias in LinearAllreduce is marked replicated, so it should appear in replicated patterns
    assert r"^proj\.bias$" in uc_info[TP_REPLICATED_PARAMETER_PATTERNS]


def test_collect_autotp_universal_checkpoint_info_subparams():
    layer = SubParamLinearLayer(torch.nn.Linear(12, 12, bias=True),
                                mp_group=None,
                                shape=(3, -1),
                                partition_dim=0,
                                name="qkv")
    model = torch.nn.Module()
    model.qkv = layer

    uc_info = collect_autotp_universal_checkpoint_info(model)

    assert [entry["patterns"] for entry in uc_info[PARAMETER_WITH_SUB_PARAMS]] == [[r"^qkv\.weight$"],
                                                                                   [r"^qkv\.bias$"]]
    assert uc_info[PARAMETER_WITH_SUB_PARAMS][0]["partition_dim"] == 0


def test_collect_autotp_universal_checkpoint_info_column_parallel_bias_not_replicated():
    layer = LinearLayer(torch.nn.Linear(16, 8, bias=True), mp_group=None, name="dense")
    model = torch.nn.Module()
    model.dense = layer

    uc_info = collect_autotp_universal_checkpoint_info(model)

    assert not any("dense.weight" in p for p in uc_info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS])
    assert not any("dense.bias" in p for p in uc_info[TP_REPLICATED_PARAMETER_PATTERNS])


def test_collect_autotp_universal_checkpoint_info_subparams_preserves_shape_metadata():
    layer = SubParamLinearLayer(torch.nn.Linear(12, 12, bias=True),
                                mp_group=None,
                                shape=((2, 10), 12),
                                partition_dim=0,
                                name="fused")
    model = torch.nn.Module()
    model.fused = layer

    uc_info = collect_autotp_universal_checkpoint_info(model)

    assert uc_info[PARAMETER_WITH_SUB_PARAMS][0]["shape"] == [(2, 10), 12]


def test_subparam_layer_marks_standardized_param_metadata():
    layer = SubParamLinearLayer(torch.nn.Linear(12, 12, bias=True),
                                mp_group=None,
                                shape=(3, -1),
                                partition_dim=0,
                                name="packed")

    weight_meta = getattr(layer.weight, DS_AUTOTP_UC_META)
    bias_meta = getattr(layer.bias, DS_AUTOTP_UC_META)

    assert weight_meta["sub_param_sizes"] == (4, 4, 4)
    assert tuple(weight_meta["target_partition_shape"]) == tuple(layer.weight.shape)
    assert tuple(bias_meta["target_partition_shape"]) == tuple(layer.bias.shape)


def test_linear_layer_marks_uneven_column_metadata():
    layer = LinearLayer(torch.nn.Linear(8, 101, bias=True), mp_group=None, name="lm_head")
    # Stand in for a layer built under tp=2; the split is normally frozen during construction.
    layer.tp_world_size = 2
    layer._freeze_partition_sizes(101)
    layer.weight.data = layer.weight.data[:51].contiguous()
    layer.bias.data = layer.bias.data[:51].contiguous()
    layer._mark_uc_metadata()

    weight_meta = getattr(layer.weight, DS_AUTOTP_UC_META)
    bias_meta = getattr(layer.bias, DS_AUTOTP_UC_META)

    assert weight_meta["logical_shape"] == (101, 8)
    assert weight_meta["output_shape"] == (101, )
    assert weight_meta["partition_sizes"] == (51, 50)
    assert weight_meta["target_partition_shape"] == (51, 8)
    assert weight_meta["original_shape"] == (101, 8)
    assert bias_meta["logical_shape"] == (101, )
    assert bias_meta["partition_sizes"] == (51, 50)
    assert bias_meta["target_partition_shape"] == (51, )


def test_universal_checkpoint_info_excludes_param_level_recovery_fields():
    layer = SubParamLinearLayer(torch.nn.Linear(12, 12, bias=True),
                                mp_group=None,
                                shape=(3, -1),
                                partition_dim=0,
                                name="packed")
    model = torch.nn.Module()
    model.packed = layer

    uc_info = collect_autotp_universal_checkpoint_info(model)
    subparam_entry = uc_info[PARAMETER_WITH_SUB_PARAMS][0]

    assert "shape" in subparam_entry
    assert "partition_dim" in subparam_entry
    assert "patterns" in subparam_entry
    assert "sub_param_sizes" not in subparam_entry
    assert "partition_sizes" not in subparam_entry
    assert "target_partition_shape" not in subparam_entry


def test_collect_uses_conversion_view_not_recovery_fields():
    layer = SubParamLinearLayer(torch.nn.Linear(12, 12, bias=True),
                                mp_group=None,
                                shape=(3, -1),
                                partition_dim=0,
                                name="packed")
    model = torch.nn.Module()
    model.packed = layer

    meta = getattr(layer.weight, "ds_autotp_universal_checkpoint_meta")
    meta["partition_dim"] = 99
    meta["sub_param_shape"] = (999, -1)

    uc_info = collect_autotp_universal_checkpoint_info(model)
    subparam_entry = uc_info[PARAMETER_WITH_SUB_PARAMS][0]

    assert subparam_entry["partition_dim"] == 0
    # The published shape carries the physical sub-parameter sizes rather than the (3, -1)
    # view spec, so a reader cannot mistake the sub-parameter count for a shard width.
    assert subparam_entry["shape"] == [(4, 4, 4), -1]


def test_collect_publishes_physical_subparam_sizes_not_the_count():
    # shape=(3, -1) is a view spec whose partition_dim entry is the sub-parameter *count*.
    # Publishing that count lets a converter read 3 as a shard width and merge the parameter
    # from a single narrow slice, so the recorded widths are used to publish real sizes.
    layer = SubParamLinearLayer(torch.nn.Linear(12, 12, bias=True),
                                mp_group=None,
                                shape=(3, -1),
                                partition_dim=0,
                                name="packed")
    model = torch.nn.Module()
    model.packed = layer

    uc_info = collect_autotp_universal_checkpoint_info(model)
    subparam_entry = uc_info[PARAMETER_WITH_SUB_PARAMS][0]

    published_sizes = subparam_entry["shape"][subparam_entry["partition_dim"]]
    assert published_sizes == (4, 4, 4)
    assert sum(published_sizes) == 12


def test_param_uc_restore_builder_normalizes_shapes_and_nests_conversion_view():
    restore_meta = _build_param_uc_restore_meta(partition_type="column",
                                                partition_dim=0,
                                                logical_shape=[12, 8],
                                                output_shape=[12],
                                                sub_param_shape=[3, -1],
                                                sub_param_sizes=[4, 4, 4],
                                                partition_sizes=[6, 6],
                                                target_partition_shape=torch.Size([4, 8]),
                                                original_shape=torch.Size([12, 8]),
                                                is_bias=False,
                                                replicated=False)

    assert restore_meta["logical_shape"] == (12, 8)
    assert restore_meta["output_shape"] == (12, )
    assert restore_meta["sub_param_shape"] == (3, -1)
    assert restore_meta["sub_param_sizes"] == (4, 4, 4)
    assert restore_meta["partition_sizes"] == (6, 6)
    assert restore_meta["target_partition_shape"] == (4, 8)
    assert restore_meta["original_shape"] == (12, 8)
    assert restore_meta["conversion"] == {
        "partition_type": "column",
        "partition_dim": 0,
        "sub_param_shape": (3, -1),
        "sub_param_shard_widths": None,
        "original_shape": (12, 8),
        "is_bias": False,
        "replicated": False,
        "unsupported_reason": None,
    }


def test_conversion_helper_reads_builder_nested_view():
    param = torch.nn.Parameter(torch.zeros(4, 8))
    param.ds_autotp_universal_checkpoint_meta = _build_param_uc_restore_meta(partition_type="row",
                                                                             partition_dim=1,
                                                                             logical_shape=[4, 16],
                                                                             output_shape=[4],
                                                                             original_shape=[4, 16],
                                                                             is_bias=False,
                                                                             replicated=False)

    assert _get_param_uc_conversion_meta(param) == param.ds_autotp_universal_checkpoint_meta["conversion"]


def test_collect_marks_autotp_unchanged_params_as_replicated():
    # AutoTP replaces the Linear (its params get conversion meta) but leaves the plain
    # LayerNorm untouched, so the LayerNorm params have no conversion meta. They must be
    # classified as TP-replicated; otherwise the converter's default dim-0 concat would
    # expand them from [H] to [H * tp_degree].
    model = torch.nn.Module()
    model.fc = LinearLayer(torch.nn.Linear(16, 8, bias=True), mp_group=None, name="fc")
    model.ln = torch.nn.LayerNorm(16)

    uc_info = collect_autotp_universal_checkpoint_info(model)

    replicated = uc_info[TP_REPLICATED_PARAMETER_PATTERNS]
    assert r"^ln\.weight$" in replicated
    assert r"^ln\.bias$" in replicated
    # The AutoTP-replaced column-parallel weight/bias are sharded, not replicated.
    assert not any("fc.weight" in p for p in replicated)
    assert not any("fc.bias" in p for p in replicated)


def test_collect_publishes_sub_param_metadata_for_partitioned_bias():
    # An uneven fused bias is cut per sub-parameter just like its weight. Without sub-parameter
    # metadata the converter concatenates the rank slices end to end, which interleaves Q/K/V
    # and silently restores a wrong bias.
    layer = SubParamLinearLayer(torch.nn.Linear(2, 12, bias=True),
                                mp_group=None,
                                shape=((6, 3, 3), -1),
                                partition_dim=0,
                                name="qkv")
    layer.tp_world_size = 2
    layer._subparam_shard_widths = ((4, 2), (2, 1), (2, 1))
    layer.weight.data = layer.weight.data[:8].contiguous()
    layer.bias.data = layer.bias.data[:8].contiguous()
    layer._mark_uc_metadata()
    model = torch.nn.Module()
    model.qkv = layer

    uc_info = collect_autotp_universal_checkpoint_info(model)

    bias_entry = next(e for e in uc_info[PARAMETER_WITH_SUB_PARAMS] if e["patterns"] == [r"^qkv\.bias$"])
    assert bias_entry["shape"] == [(6, 3, 3)]
    assert bias_entry["partition_dim"] == 0
    assert uc_info[SUB_PARAM_SHARD_WIDTHS][r"^qkv\.bias$"] == [[4, 2], [2, 1], [2, 1]]


def test_collect_skips_tied_parameter_aliases():
    # Both names reach the same tensor, but the optimizer only knows the first one, so a
    # pattern for the alias would describe a parameter that has no TP slices to convert.
    layer = LinearLayer(torch.nn.Linear(8, 16, bias=False), mp_group=None, name="embed_tokens")
    model = torch.nn.Module()
    model.embed_tokens = layer
    model.lm_head = torch.nn.Linear(8, 16, bias=False)
    model.lm_head.weight = layer.weight

    uc_info = collect_autotp_universal_checkpoint_info(model)

    all_patterns = (uc_info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS] + uc_info[TP_REPLICATED_PARAMETER_PATTERNS] +
                    uc_info[VOCABULARY_PARAMETER_PATTERNS] +
                    [p for entry in uc_info[PARAMETER_WITH_SUB_PARAMS] for p in entry["patterns"]])
    assert not any("lm_head" in pattern for pattern in all_patterns)


def test_collect_rejects_fused_layouts_without_a_sub_param_split():
    # codegen interleaves the Q/K/V blocks across ranks, so no per-sub-parameter description
    # exists. Publishing a plain column layout would reassemble the weight in rank order.
    class CodeGenBlock(torch.nn.Module):
        pass

    fused_module = CodeGenBlock()
    layer = fused_LinearLayer(torch.nn.Linear(8, 12, bias=False),
                              mp_group=None,
                              skip_partition=True,
                              fused_module=fused_module,
                              name="qkv_proj")
    model = torch.nn.Module()
    model.qkv_proj = layer

    with pytest.raises(ValueError, match="qkv_proj.weight"):
        collect_autotp_universal_checkpoint_info(model)


def test_gate_up_pack_publishes_sub_param_metadata():
    layer = GateUpPack_LinearLayer(torch.nn.Linear(4, 12, bias=False), mp_group=None, name="dense_h_to_4h")
    layer.tp_world_size = 2
    layer._freeze_partition_sizes(12)
    layer.weight.data = layer.weight.data[:6].contiguous()
    layer._mark_uc_metadata()
    model = torch.nn.Module()
    model.dense_h_to_4h = layer

    uc_info = collect_autotp_universal_checkpoint_info(model)

    entry = uc_info[PARAMETER_WITH_SUB_PARAMS][0]
    assert entry["shape"] == [(6, 6), 4]
    assert uc_info[SUB_PARAM_SHARD_WIDTHS][r"^dense_h_to_4h\.weight$"] == [[3, 3], [3, 3]]
