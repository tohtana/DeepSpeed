# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from unit.common import DistributedTest, preferred_dtype
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.auto_tp import AutoTP
from deepspeed.module_inject.autotp_config import AutoTPConfig, AutoTPPresets, PartitionType
from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer, SubParamLinearLayer
from deepspeed.runtime.tensor_parallel.config import TPTrainingConfig
from deepspeed.utils import groups


def skip_on_device():
    if get_accelerator().device_name() == "xpu":
        pytest.skip("XPU requires a higher version for test")


def assert_close_for_preferred_dtype(actual, expected):
    atol = 3e-3
    rtol = 2e-2
    if preferred_dtype() is torch.float32:
        atol = 1e-5
        rtol = 1e-5
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def make_mock_qwen35_config():
    return SimpleNamespace(
        model_type="qwen3_5_text",
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_num_value_heads=2,
        linear_value_head_dim=8,
    )


class MockLinearAttention(nn.Module):
    """GatedDeltaNet-style linear-attention submodule."""

    linear_num_key_heads = 2
    linear_key_head_dim = 4
    linear_num_value_heads = 2
    linear_value_head_dim = 8

    def __init__(self, hidden_dim):
        super().__init__()
        self.in_proj_qkv = nn.Linear(hidden_dim, self.q_size + self.k_size + self.v_size)
        self.in_proj_z = nn.Linear(hidden_dim, self.q_size)
        self.in_proj_a = nn.Linear(hidden_dim, hidden_dim)
        self.in_proj_b = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(self.q_size + self.k_size + self.v_size, hidden_dim)

    @property
    def q_size(self):
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def k_size(self):
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def v_size(self):
        return self.linear_num_value_heads * self.linear_value_head_dim

    def forward(self, x):
        qkv = self.in_proj_qkv(x)
        q, k, v = torch.split(qkv, [self.q_size, self.k_size, self.v_size], dim=-1)
        z = torch.sigmoid(self.in_proj_z(x))
        mixed = torch.cat((q * z, k, v), dim=-1)
        return self.out_proj(mixed)


class MockFullAttention(nn.Module):
    """Standard multi-head attention submodule."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class MockGatedQProjAttention(nn.Module):
    """Full-attention block with a widened q_proj output."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.o_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        return self.o_proj(self.q_proj(x))


class MockMLP(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.down_proj = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class MockLinearAttnDecoderLayer(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.linear_attn = MockLinearAttention(hidden_dim)
        self.mlp = MockMLP(hidden_dim)

    def forward(self, x):
        return self.mlp(self.linear_attn(x))


class MockFullAttnDecoderLayer(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = MockFullAttention(hidden_dim)
        self.mlp = MockMLP(hidden_dim)

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class MockGatedQProjLayer(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = MockGatedQProjAttention(hidden_dim)

    def forward(self, x):
        return self.self_attn(x)


class MockQwen35HybridModel(nn.Module):
    """4-layer hybrid model with full attention every fourth layer."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.config = make_mock_qwen35_config()
        self.layers = nn.ModuleList([
            MockLinearAttnDecoderLayer(hidden_dim),
            MockLinearAttnDecoderLayer(hidden_dim),
            MockLinearAttnDecoderLayer(hidden_dim),
            MockFullAttnDecoderLayer(hidden_dim),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MockGatedQProjModel(nn.Module):
    """Single-layer model for widened q_proj sharding checks."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.config = make_mock_qwen35_config()
        self.layers = nn.ModuleList([MockGatedQProjLayer(hidden_dim)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestQwen35PresetPatterns:
    """Verify qwen3_5 preset resolution and pattern coverage."""

    def test_get_preset_returns_config(self):
        config = AutoTPPresets.get_preset("qwen3_5")
        assert config is not None
        assert isinstance(config, AutoTPConfig)

    def test_preset_has_seven_layer_specs(self):
        config = AutoTPPresets.get_preset("qwen3_5")
        assert len(config.layer_specs) == 7

    def test_self_attn_column_parallel(self):
        config = AutoTPPresets.get_preset("qwen3_5")
        for proj in ["q_proj", "k_proj", "v_proj"]:
            spec = config.find_matching_spec(f"model.layers.3.self_attn.{proj}.weight")
            assert spec is not None, f"self_attn.{proj} should match"
            assert spec.partition_type == PartitionType.COLUMN

    def test_self_attn_row_parallel(self):
        config = AutoTPPresets.get_preset("qwen3_5")
        spec = config.find_matching_spec("model.layers.3.self_attn.o_proj.weight")
        assert spec is not None
        assert spec.partition_type == PartitionType.ROW

    def test_mlp_matches_in_linear_attn_layer(self):
        config = AutoTPPresets.get_preset("qwen3_5")

        spec = config.find_matching_spec("model.layers.0.mlp.gate_proj.weight")
        assert spec is not None
        assert spec.partition_type == PartitionType.COLUMN

        spec = config.find_matching_spec("model.layers.0.mlp.up_proj.weight")
        assert spec is not None
        assert spec.partition_type == PartitionType.COLUMN

        spec = config.find_matching_spec("model.layers.0.mlp.down_proj.weight")
        assert spec is not None
        assert spec.partition_type == PartitionType.ROW

    def test_linear_attn_supported_linears_match(self):
        config = AutoTPPresets.get_preset("qwen3_5")

        qkv_spec = config.find_matching_spec("model.layers.0.linear_attn.in_proj_qkv.weight")
        assert qkv_spec is not None
        assert qkv_spec.partition_type == PartitionType.COLUMN
        assert qkv_spec.shape_resolver == "qwen3_5_linear_attn_qkv"

        z_spec = config.find_matching_spec("model.layers.0.linear_attn.in_proj_z.weight")
        assert z_spec is not None
        assert z_spec.partition_type == PartitionType.COLUMN

        out_spec = config.find_matching_spec("model.layers.0.linear_attn.out_proj.weight")
        assert out_spec is not None
        assert out_spec.partition_type == PartitionType.ROW

    def test_linear_attn_remaining_weights_not_matched(self):
        config = AutoTPPresets.get_preset("qwen3_5")
        unmatched_names = [
            "model.layers.0.linear_attn.in_proj_a.weight",
            "model.layers.0.linear_attn.in_proj_b.weight",
            "model.layers.0.linear_attn.conv1d.weight",
            "model.layers.0.linear_attn.dt_bias",
            "model.layers.0.linear_attn.A_log",
        ]
        for name in unmatched_names:
            assert config.find_matching_spec(name) is None, f"Preset should not match {name}"

    def test_moe_names_not_matched(self):
        config = AutoTPPresets.get_preset("qwen3_5")
        moe_names = [
            "model.layers.0.mlp.experts.gate_up_proj.weight",
            "model.layers.0.mlp.experts.down_proj.weight",
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.layers.0.mlp.shared_expert.down_proj.weight",
        ]
        for name in moe_names:
            assert config.find_matching_spec(name) is None, f"Dense preset should not match {name}"

    def test_multimodal_prefix_still_matches(self):
        config = AutoTPPresets.get_preset("qwen3_5")

        spec = config.find_matching_spec("model.language_model.layers.3.self_attn.q_proj.weight")
        assert spec is not None
        assert spec.partition_type == PartitionType.COLUMN

        spec = config.find_matching_spec("model.language_model.layers.0.mlp.down_proj.weight")
        assert spec is not None
        assert spec.partition_type == PartitionType.ROW

    def test_preset_via_get_partition_config_object(self):
        tp_config = TPTrainingConfig(autotp_size=2, preset_model="qwen3_5")
        config = tp_config.get_partition_config_object()
        assert config is not None
        assert config.tp_size == 2
        assert len(config.layer_specs) == 7


class TestQwen35MockHybridModel(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def _apply_preset(self, model, strict_mode=False):
        groups._init_tp_mesh_device(tensor_model_parallel_size=2)
        config = AutoTPPresets.get_preset("qwen3_5")
        config.strict_mode = strict_mode
        autotp = AutoTP(
            module=model,
            all_reduce_linears=[],
            prefix="",
            state_dict=None,
            linear_layer_setting=None,
            orig_layer_impl=None,
            keep_module_on_host=False,
            partition_config=config,
        )
        autotp.set_tensor_parallel_config(2, groups.get_tensor_model_parallel_group())
        autotp.update_linear_policies()
        autotp._replace_module(model)
        return model

    def test_preset_replaces_supported_layers(self):
        skip_on_device()
        model = MockQwen35HybridModel(hidden_dim=16)
        model = self._apply_preset(model)

        assert isinstance(model.layers[3].self_attn.q_proj, LinearLayer)
        assert isinstance(model.layers[3].self_attn.k_proj, LinearLayer)
        assert isinstance(model.layers[3].self_attn.v_proj, LinearLayer)
        assert isinstance(model.layers[3].self_attn.o_proj, LinearAllreduce)

        assert isinstance(model.layers[3].mlp.gate_proj, LinearLayer)
        assert isinstance(model.layers[3].mlp.up_proj, LinearLayer)
        assert isinstance(model.layers[3].mlp.down_proj, LinearAllreduce)

        for i in range(3):
            assert isinstance(model.layers[i].mlp.gate_proj, LinearLayer)
            assert isinstance(model.layers[i].mlp.up_proj, LinearLayer)
            assert isinstance(model.layers[i].mlp.down_proj, LinearAllreduce)

            assert isinstance(model.layers[i].linear_attn.in_proj_qkv, SubParamLinearLayer)
            assert isinstance(model.layers[i].linear_attn.in_proj_z, LinearLayer)
            assert isinstance(model.layers[i].linear_attn.out_proj, LinearAllreduce)
            assert isinstance(model.layers[i].linear_attn.in_proj_a, nn.Linear)
            assert isinstance(model.layers[i].linear_attn.in_proj_b, nn.Linear)
            assert model.layers[i].linear_attn.in_proj_qkv.weight.shape == (16, 16)

    def test_strict_mode_raises_for_remaining_linear_attn_weights(self):
        skip_on_device()
        model = MockQwen35HybridModel(hidden_dim=16)
        with pytest.raises(ValueError, match=r"No matching spec for .*linear_attn\.in_proj_a\.weight"):
            self._apply_preset(model, strict_mode=True)

    def test_linear_attn_supported_weights_shard_cleanly(self):
        skip_on_device()
        hidden_dim = 16
        device = get_accelerator().current_device_name()

        torch.manual_seed(1234)
        model = MockQwen35HybridModel(hidden_dim).to(device=device, dtype=preferred_dtype())
        baseline = deepcopy(model)
        model = self._apply_preset(model)

        torch.manual_seed(4321)
        inputs = torch.randn(2, hidden_dim, dtype=preferred_dtype(), device=device)
        full_output = baseline(inputs)
        tp_output = model(inputs)
        assert_close_for_preferred_dtype(tp_output, full_output)

    def test_gated_q_proj_output_width_still_shards_cleanly(self):
        skip_on_device()
        hidden_dim = 16
        device = get_accelerator().current_device_name()

        torch.manual_seed(1234)
        model = MockGatedQProjModel(hidden_dim).to(device=device, dtype=preferred_dtype())
        baseline = deepcopy(model)
        model = self._apply_preset(model)

        q_proj = model.layers[0].self_attn.q_proj
        o_proj = model.layers[0].self_attn.o_proj

        assert isinstance(q_proj, LinearLayer)
        assert isinstance(o_proj, LinearAllreduce)
        assert q_proj.weight.shape == (hidden_dim, hidden_dim)

        torch.manual_seed(4321)
        inputs = torch.randn(2, hidden_dim, dtype=preferred_dtype(), device=device)
        full_output = baseline(inputs)
        tp_output = model(inputs)
        assert_close_for_preferred_dtype(tp_output, full_output)
