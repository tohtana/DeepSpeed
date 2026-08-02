# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Pipeline-parallel AutoEP ZeRO-1 Universal Checkpoint coverage."""

import glob
import json
import os
import re
from types import SimpleNamespace

import deepspeed
import deepspeed.comm as dist
import pytest
import torch
import torch.nn as nn

from deepspeed.checkpoint.constants import (
    AUTOEP_LAYERS_KEY,
    CAT_DIM,
    CHECKPOINT_PARALLEL_DIMS,
    CHECKPOINT_PP_DEGREE,
    CHECKPOINT_TP_DEGREE,
    EP_IS_EXPERT_PARAM,
    EP_NUM_EXPERTS,
    EXPERT_PARAMETER_PATTERNS,
    PARAM,
    UNIVERSAL_CHECKPOINT_INFO,
)
from deepspeed.checkpoint.deepspeed_checkpoint import DeepSpeedCheckpoint
from deepspeed.checkpoint.ds_to_universal import main as convert_to_universal
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.runtime.pipe.module import PipelineModule
from deepspeed.runtime.pipe.topology import PipeDataParallelTopology
from deepspeed.utils import RepeatingLoader
from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import (
    MockHFConfig,
    MockMoEBlock,
    make_autoep_config,
    seed_everything,
    skip_unless_h100_tests_enabled,
)

EXPERT_STATE_KEYS = ("fp32", "exp_avg", "exp_avg_sq")
EXPECTED_LAYER_PATHS = ("0.mlp", "2.mlp")


class _MoEPipeBlock(nn.Module):

    def __init__(self, num_experts, hidden_size, intermediate_size):
        super().__init__()
        self.mlp = MockMoEBlock(num_experts=num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(self, hidden_states):
        return self.mlp(hidden_states)


class _DensePipeBlock(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return torch.tanh(self.proj(hidden_states))


def _make_pipeline_engine(num_experts=4,
                          hidden_size=64,
                          intermediate_size=128,
                          load_universal=False,
                          use_data_before_expert_parallelism=False):
    topology = PipeDataParallelTopology(num_pp=2, num_dp=2)
    layers = [
        _MoEPipeBlock(num_experts, hidden_size, intermediate_size),
        _DensePipeBlock(hidden_size),
        _MoEPipeBlock(num_experts, hidden_size, intermediate_size),
        _DensePipeBlock(hidden_size),
    ]
    model = PipelineModule(layers=layers, topology=topology, loss_fn=nn.MSELoss(), partition_method="uniform")
    model.config = MockHFConfig()
    model.config.num_local_experts = num_experts
    model.config.hidden_size = hidden_size
    model.config.intermediate_size = intermediate_size

    config = make_autoep_config(zero_stage=1, ep_size=2, mixed_precision=True)
    config["expert_parallel"]["moe_layer_pattern"] = r"(?:0|2)\.mlp"
    config["use_data_before_expert_parallelism"] = use_data_before_expert_parallelism
    config["pipeline"] = {"activation_checkpoint_interval": 0}
    if load_universal:
        config["checkpoint"] = {"load_universal": True}

    engine, _, _, _ = deepspeed.initialize(model=model, config=config)
    assert isinstance(engine, PipelineEngine)
    return engine


def _set_repeating_data(engine, seed):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(1, 8, 64, generator=generator)
    labels = torch.randn(1, 8, 64, generator=generator)
    engine.set_dataiterator(RepeatingLoader([(inputs, labels)]))


def _train_batches(engine, count, seed):
    _set_repeating_data(engine, seed)
    losses = []
    for _ in range(count):
        loss = engine.train_batch()
        if torch.is_tensor(loss):
            assert torch.isfinite(loss).all()
            losses.append(float(loss.detach().float().cpu()))
    return losses


def _expert_params(engine):
    for module_name, module in engine.module.named_modules():
        if not isinstance(module, AutoEPMoELayer):
            continue
        prefix = f"{module_name}." if module_name else ""
        for weight_name in ("w1", "w2", "w3"):
            yield f"{prefix}experts.{weight_name}", module, getattr(module.experts, weight_name)


def _clone_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def _clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    return value


def _values_equal(left, right):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    return left == right


def _nonexpert_model_state(engine):
    expert_names = {param_name for param_name, _, _ in _expert_params(engine)}
    return {
        param_name: param.detach().cpu().clone()
        for param_name, param in engine.module.named_parameters() if param_name not in expert_names
    }


def _local_optimizer_state(engine):
    state = []
    for flat_param in engine.optimizer.single_partition_of_fp32_groups:
        state.append({
            "fp32": flat_param.detach().cpu().clone(),
            "adam": _clone_nested(engine.optimizer.optimizer.state.get(flat_param, {})),
        })
    return state


def _assert_nested_exact(actual, expected, path="state"):
    assert type(actual) is type(expected), f"{path}: {type(actual)} != {type(expected)}"
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"{path}: key mismatch"
        for key in expected:
            _assert_nested_exact(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), f"{path}: length mismatch"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_exact(actual_item, expected_item, f"{path}[{index}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _optimizer_steps(engine):
    steps = []
    for flat_param in engine.optimizer.single_partition_of_fp32_groups:
        value = engine.optimizer.optimizer.state[flat_param]["step"]
        steps.append(int(value.item()) if torch.is_tensor(value) else int(value))
    return steps


def _snapshot_expected_expert_states(engine, num_experts):
    local_records = []
    local_errors = []
    zero_dp_rank = dist.get_rank(group=engine.optimizer.dp_process_group)
    for param_name, module, param in _expert_params(engine):
        mapping = getattr(param, "_hp_mapping", None)
        if mapping is None:
            local_errors.append(f"{param_name}: missing ZeRO high-precision mapping")
            continue

        lp_address = mapping.lp_fragment_address
        hp_address = mapping.hp_fragment_address
        if lp_address.numel != hp_address.numel:
            local_errors.append(
                f"{param_name}: LP/HP fragment sizes differ ({lp_address.numel} != {hp_address.numel})")
            continue

        flat_param = engine.optimizer.single_partition_of_fp32_groups[mapping.param_group_index]
        optimizer_state = engine.optimizer.optimizer.state.get(flat_param, {})
        record = {
            "param_name": param_name,
            "stage_id": engine.stage_id,
            "zero_dp_rank": zero_dp_rank,
            "ep_rank": module.ep_rank,
            "ep_size": module.ep_size,
            "local_shape": tuple(param.shape),
            "start": lp_address.start,
            "numel": lp_address.numel,
            "states": {
                "fp32": flat_param.narrow(0, hp_address.start, hp_address.numel).detach().float().cpu().clone(),
            },
        }
        for state_key in ("exp_avg", "exp_avg_sq"):
            if state_key not in optimizer_state:
                local_errors.append(f"{param_name}/{state_key}: missing from underlying Adam state")
                continue
            record["states"][state_key] = optimizer_state[state_key].narrow(
                0, hp_address.start, hp_address.numel).detach().float().cpu().clone()
        if "step" not in optimizer_state:
            local_errors.append(f"{param_name}/step: missing from underlying Adam state")
        else:
            record["step"] = _clone_value(optimizer_state["step"])
        local_records.append(record)

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, {"records": local_records, "errors": local_errors})
    errors = [error for payload in gathered for error in payload["errors"]]
    assert not errors, f"failed to snapshot source expert optimizer fragments: {errors}"

    expected = {}
    coverage_summary = {}
    nonzero_moments = {"exp_avg": False, "exp_avg_sq": False}
    records = [record for payload in gathered for record in payload["records"]]
    expected_param_names = {
        f"{layer_path}.experts.{weight_name}"
        for layer_path in EXPECTED_LAYER_PATHS
        for weight_name in ("w1", "w2", "w3")
    }
    assert {record["param_name"] for record in records} == expected_param_names

    for param_name in sorted(expected_param_names):
        param_records = [record for record in records if record["param_name"] == param_name]
        ep_sizes = {record["ep_size"] for record in param_records}
        local_shapes = {record["local_shape"] for record in param_records}
        stage_ids = {record["stage_id"] for record in param_records}
        assert len(ep_sizes) == 1
        assert len(local_shapes) == 1
        assert stage_ids == {int(param_name.split(".", 1)[0]) // 2}
        ep_size = ep_sizes.pop()
        local_shape = local_shapes.pop()
        assert local_shape[0] * ep_size == num_experts

        expected[param_name] = {"states": {}}
        coverage_summary[param_name] = {}
        for state_key in EXPERT_STATE_KEYS:
            ep_tensors = []
            for ep_rank in range(ep_size):
                tensor = torch.zeros(local_shape, dtype=torch.float32).flatten()
                coverage = torch.zeros(tensor.numel(), dtype=torch.int64)
                for record in param_records:
                    if record["ep_rank"] != ep_rank or state_key not in record["states"]:
                        continue
                    start = record["start"]
                    numel = record["numel"]
                    tensor.narrow(0, start, numel).copy_(record["states"][state_key].flatten())
                    coverage.narrow(0, start, numel).add_(1)
                assert torch.all(coverage == 1), (
                    f"{param_name}/{state_key}/ep_rank={ep_rank}: source fragment coverage is not exactly one")
                coverage_summary[param_name][f"{state_key}/ep_rank={ep_rank}"] = {
                    "min": int(coverage.min()),
                    "max": int(coverage.max()),
                    "numel": coverage.numel(),
                }
                ep_tensors.append(tensor.reshape(local_shape))
            expected[param_name]["states"][state_key] = torch.cat(ep_tensors, dim=0)
            state = expected[param_name]["states"][state_key]
            coverage_summary[param_name][f"{state_key}/tensor"] = {
                "shape": list(state.shape),
                "dtype": str(state.dtype),
                "nonzero": int(torch.count_nonzero(state)),
            }
            if state_key != "fp32" and torch.count_nonzero(state).item() > 0:
                nonzero_moments[state_key] = True

        step_values = [record["step"] for record in param_records if "step" in record]
        assert step_values
        assert all(_values_equal(step_values[0], value) for value in step_values[1:])
        expected[param_name]["step"] = step_values[0]
        step = step_values[0].item() if torch.is_tensor(step_values[0]) else step_values[0]
        assert step > 0
        coverage_summary[param_name]["step"] = step
    assert nonzero_moments == {"exp_avg": True, "exp_avg_sq": True}
    coverage_summary["nonzero_moments"] = nonzero_moments
    return expected, coverage_summary


def _runtime_metadata(engine):
    replacements = [(name, module.ep_rank, module.ep_size, module.num_local_experts)
                    for name, module in engine.module.named_modules() if isinstance(module, AutoEPMoELayer)]
    optimizer_dp_global_ranks = [
        dist.get_global_rank(engine.optimizer.dp_process_group, rank)
        for rank in range(dist.get_world_size(group=engine.optimizer.dp_process_group))
    ]
    ep_global_ranks = []
    if replacements:
        autoep_module = next(module for module in engine.module.modules() if isinstance(module, AutoEPMoELayer))
        ep_global_ranks = [
            dist.get_global_rank(autoep_module.ep_group, rank)
            for rank in range(dist.get_world_size(group=autoep_module.ep_group))
        ]
    local = {
        "global_rank": dist.get_rank(),
        "stage_id": engine.stage_id,
        "local_bounds": [engine.module._local_start, engine.module._local_stop],
        "model_parallel_rank": engine.mpu.get_model_parallel_rank(),
        "pipe_parallel_size": engine.mpu.get_pipe_parallel_world_size(),
        "data_parallel_size": engine.mpu.get_data_parallel_world_size(),
        "tensor_parallel_size": engine.mpu.get_slice_parallel_world_size(),
        "zero_dp_rank": dist.get_rank(group=engine.optimizer.dp_process_group),
        "zero_dp_world_size": dist.get_world_size(group=engine.optimizer.dp_process_group),
        "optimizer_dp_global_ranks": optimizer_dp_global_ranks,
        "ep_global_ranks": ep_global_ranks,
        "replacements": replacements,
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    return gathered


def _assert_runtime_metadata(metadata):
    for stage_id in (0, 1):
        stage = [entry for entry in metadata if entry["stage_id"] == stage_id]
        assert len(stage) == 2
        assert {entry["zero_dp_rank"] for entry in stage} == {0, 1}
        assert {entry["zero_dp_world_size"] for entry in stage} == {2}
        assert {tuple(entry["optimizer_dp_global_ranks"]) for entry in stage} == {(stage_id * 2, stage_id * 2 + 1)}
        assert {tuple(entry["ep_global_ranks"]) for entry in stage} == {(stage_id * 2, stage_id * 2 + 1)}
        assert {entry["model_parallel_rank"] for entry in stage} == {stage_id}
        assert {tuple(entry["local_bounds"]) for entry in stage} == {(stage_id * 2, stage_id * 2 + 2)}
        assert {entry["pipe_parallel_size"] for entry in stage} == {2}
        assert {entry["data_parallel_size"] for entry in stage} == {2}
        assert {entry["tensor_parallel_size"] for entry in stage} == {1}
        assert {len(entry["replacements"]) for entry in stage} == {1}
        assert {entry["replacements"][0][0] for entry in stage} == {f"{stage_id * 2}.mlp"}
        assert {entry["replacements"][0][1] for entry in stage} == {0, 1}
        assert {entry["replacements"][0][2] for entry in stage} == {2}
        assert {entry["replacements"][0][3] for entry in stage} == {2}


def _metadata_paths(state):
    metadata = state.get(AUTOEP_LAYERS_KEY)
    if metadata is None:
        return []
    assert isinstance(metadata, list)
    return sorted(entry["module_path"] for entry in metadata)


def _rank_zero_call(function):
    payload = [None]
    if dist.get_rank() == 0:
        try:
            payload[0] = {"result": function(), "error": None}
        except Exception as exc:
            payload[0] = {"result": None, "error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(payload, src=0)
    if payload[0]["error"] is not None:
        raise AssertionError(payload[0]["error"])
    return payload[0]["result"]


def _inspect_source_checkpoint(checkpoint_dir):
    assert not glob.glob(os.path.join(checkpoint_dir, "expp_rank_*_optim_states.pt")), (
        "ZeRO-1/2 checkpoints must use authoritative ZeRO optimizer shards instead of empty per-expert payloads")
    model_files = sorted(glob.glob(os.path.join(checkpoint_dir, "mp_rank_*_model_states.pt")))
    assert [os.path.basename(path) for path in model_files] == [
        "mp_rank_00_model_states.pt",
        "mp_rank_01_model_states.pt",
    ]

    metadata_by_stage = {}
    metadata_entries = {}
    parallel_dimensions_by_stage = {}
    for path in model_files:
        stage_id = int(re.search(r"mp_rank_(\d+)_", os.path.basename(path)).group(1))
        state = torch.load(path, map_location="cpu", weights_only=False)
        metadata_by_stage[stage_id] = _metadata_paths(state)
        metadata_entries[stage_id] = state.get(AUTOEP_LAYERS_KEY)
        parallel_dimensions_by_stage[stage_id] = state.get(CHECKPOINT_PARALLEL_DIMS)
    assert metadata_by_stage == {0: ["0.mlp"], 1: ["2.mlp"]}
    assert parallel_dimensions_by_stage == {
        0: {
            CHECKPOINT_PP_DEGREE: 2,
            CHECKPOINT_TP_DEGREE: 1,
        },
        1: {
            CHECKPOINT_PP_DEGREE: 2,
            CHECKPOINT_TP_DEGREE: 1,
        },
    }
    for stage_id, expected_path in enumerate(EXPECTED_LAYER_PATHS):
        entries = metadata_entries[stage_id]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["moe_layer_id"] == 0
        assert entry["module_path"] == expected_path
        assert entry["expert_key_prefix"] == f"{expected_path}.experts"
        assert entry["num_experts"] == 4
        assert entry["num_local_experts"] == 2
        assert entry["ep_size"] == 2

    zero_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*zero_pp_rank_*_optim_states.pt")))
    zero_ranks_by_stage = {0: [], 1: []}
    for path in zero_files:
        match = re.fullmatch(r"(?:bf16_)?zero_pp_rank_(\d+)_mp_rank_(\d+)_optim_states\.pt", os.path.basename(path))
        assert match is not None, os.path.basename(path)
        zero_ranks_by_stage[int(match.group(2))].append(int(match.group(1)))
    assert {stage: sorted(ranks) for stage, ranks in zero_ranks_by_stage.items()} == {0: [0, 1], 1: [0, 1]}

    discovered = DeepSpeedCheckpoint(checkpoint_dir)

    return {
        "model_state_files":
        sorted(os.path.basename(path) for path in glob.glob(os.path.join(checkpoint_dir, "*model_states.pt"))),
        "metadata_by_stage":
        metadata_by_stage,
        "parallel_dimensions_by_stage":
        parallel_dimensions_by_stage,
        "zero_files": [os.path.basename(path) for path in zero_files],
        "zero_ranks_by_stage":
        zero_ranks_by_stage,
        "discovered_topology": {
            "pp_degree": discovered.pp_degree,
            "dp_degree": discovered.dp_degree,
            "tp_degree": discovered.tp_degree,
        },
    }


def _convert_checkpoint(save_dir, tag):
    checkpoint_dir = os.path.join(save_dir, tag)
    universal_dir = os.path.join(save_dir, f"{tag}_universal")
    args = SimpleNamespace(input_folder=checkpoint_dir,
                           output_folder=universal_dir,
                           num_extract_workers=1,
                           num_merge_workers=1,
                           keep_temp_folder=False,
                           strict=True,
                           inject_missing_state=False)

    conversion_error = [None]
    if dist.get_rank() == 0:
        try:
            convert_to_universal(args)
        except Exception as exc:
            conversion_error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(conversion_error, src=0)
    if conversion_error[0] is not None:
        raise RuntimeError(f"AutoEP Pipeline Universal Checkpoint conversion failed: {conversion_error[0]}")
    return universal_dir


def _inspect_universal_checkpoint(universal_dir, expected):
    summary = {"model_metadata": {}, "expert_states": {}, "problems": []}
    for path in sorted(glob.glob(os.path.join(universal_dir, "mp_rank_*_model_states.pt"))):
        basename = os.path.basename(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        universal_info = state.get(UNIVERSAL_CHECKPOINT_INFO, {})
        metadata = universal_info.get(AUTOEP_LAYERS_KEY)
        expert_patterns = universal_info.get(EXPERT_PARAMETER_PATTERNS)
        paths = sorted(entry["module_path"] for entry in metadata) if isinstance(metadata, list) else []
        summary["model_metadata"][basename] = {"paths": paths, "expert_patterns": expert_patterns}
        if paths != list(EXPECTED_LAYER_PATHS):
            summary["problems"].append(
                f"{basename}: universal AutoEP metadata {paths} != {list(EXPECTED_LAYER_PATHS)}")
        if not isinstance(expert_patterns, list) or not expert_patterns:
            summary["problems"].append(f"{basename}: missing universal expert parameter pattern")

    expected_model_files = {"mp_rank_00_model_states.pt", "mp_rank_01_model_states.pt"}
    if set(summary["model_metadata"]) != expected_model_files:
        summary["problems"].append(
            f"universal model files {sorted(summary['model_metadata'])} != {sorted(expected_model_files)}")

    nonzero_moments = {"exp_avg": False, "exp_avg_sq": False}
    for param_name, expected_state in expected.items():
        param_summary = {}
        param_dir = os.path.join(universal_dir, "zero", param_name)
        for state_key in EXPERT_STATE_KEYS:
            path = os.path.join(param_dir, f"{state_key}.pt")
            state_summary = {"exists": os.path.isfile(path)}
            if os.path.isfile(path):
                state = torch.load(path, map_location="cpu", weights_only=False)
                actual = state.get(PARAM)
                state_summary.update({
                    "shape":
                    list(actual.shape) if torch.is_tensor(actual) else None,
                    "dtype":
                    str(actual.dtype) if torch.is_tensor(actual) else None,
                    "is_expert_param":
                    state.get(EP_IS_EXPERT_PARAM),
                    "ep_num_experts":
                    state.get(EP_NUM_EXPERTS),
                    "cat_dim":
                    state.get(CAT_DIM),
                    "exact":
                    torch.is_tensor(actual) and torch.equal(actual, expected_state["states"][state_key]),
                })
                if state_key != "fp32" and torch.is_tensor(actual) and torch.count_nonzero(actual).item() > 0:
                    nonzero_moments[state_key] = True
            param_summary[state_key] = state_summary
            if not (state_summary.get("exists") and state_summary.get("is_expert_param") is True
                    and state_summary.get("ep_num_experts") == 4 and state_summary.get("cat_dim") == 0
                    and state_summary.get("dtype") == "torch.float32" and state_summary.get("shape", [0])[0] == 4
                    and state_summary.get("exact") is True):
                summary["problems"].append(f"{param_name}/{state_key}: {state_summary}")

        step_path = os.path.join(param_dir, "step.pt")
        step_summary = {"exists": os.path.isfile(step_path)}
        if os.path.isfile(step_path):
            actual_step = torch.load(step_path, map_location="cpu", weights_only=False)
            step_summary["exact"] = _values_equal(actual_step, expected_state["step"])
            step_summary["value"] = actual_step.tolist() if torch.is_tensor(actual_step) else actual_step
        param_summary["step"] = step_summary
        if not (step_summary.get("exists") and step_summary.get("exact") is True):
            summary["problems"].append(f"{param_name}/step: {step_summary}")
        summary["expert_states"][param_name] = param_summary

    if nonzero_moments != {"exp_avg": True, "exp_avg_sq": True}:
        summary["problems"].append(f"nonzero Adam moments not observed: {nonzero_moments}")
    summary["nonzero_moments"] = nonzero_moments
    return summary


def _assert_expert_model_weights(engine, expected):
    mismatches = []
    for param_name, module, param in _expert_params(engine):
        full_state = expected[param_name]["states"]["fp32"]
        local_experts = full_state.shape[0] // module.ep_size
        start = module.ep_rank * local_experts
        expected_local = full_state[start:start + local_experts].to(device=param.device, dtype=param.dtype)
        if not torch.equal(param.detach(), expected_local):
            max_diff = (param.detach().float() - expected_local.float()).abs().max().item()
            mismatches.append(f"{param_name}: loaded model weight differs (max abs diff {max_diff})")
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, mismatches)
    assert not any(gathered), f"loaded expert model-weight mismatches by rank: {gathered}"


def _assert_snapshots_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for param_name in expected:
        for state_key in EXPERT_STATE_KEYS:
            torch.testing.assert_close(actual[param_name]["states"][state_key],
                                       expected[param_name]["states"][state_key],
                                       rtol=0,
                                       atol=0)
        assert _values_equal(actual[param_name]["step"], expected[param_name]["step"])


class TestAutoEPZero1UniversalCheckpointPipeline(DistributedTest):
    world_size = 4

    @pytest.mark.parametrize("use_data_before_expert_parallelism", [False, True], ids=["E+D", "D+E"])
    def test_save_convert_load_preserves_stage_local_expert_state(self, tmpdir, use_data_before_expert_parallelism):
        skip_unless_h100_tests_enabled("AutoEP ZeRO-1 Pipeline Universal Checkpoint coverage requires H100")
        seed_everything(8198)
        topology_name = "de" if use_data_before_expert_parallelism else "ed"
        save_dir = os.path.join(str(tmpdir), f"autoep-zero1-pipeline-{topology_name}")
        tag = f"autoep-zero1-pp2-{topology_name}"
        engine = None
        reloaded = None
        try:
            engine = _make_pipeline_engine(use_data_before_expert_parallelism=use_data_before_expert_parallelism)
            source_runtime = _runtime_metadata(engine)
            _assert_runtime_metadata(source_runtime)
            losses = _train_batches(engine, count=2, seed=8198)
            expected, coverage = _snapshot_expected_expert_states(engine, num_experts=4)
            expected_nonexpert = _nonexpert_model_state(engine)
            expected_optimizer = _local_optimizer_state(engine)

            engine.save_checkpoint(save_dir, tag=tag)
            source_checkpoint = _rank_zero_call(lambda: _inspect_source_checkpoint(os.path.join(save_dir, tag)))
            if dist.get_rank() == 0:
                print("AUTOEP_PP2_SOURCE=" + json.dumps(
                    {
                        "runtime": source_runtime,
                        "losses": losses,
                        "coverage": coverage,
                        "checkpoint": source_checkpoint,
                    },
                    sort_keys=True))
            assert source_checkpoint["discovered_topology"] == {
                "pp_degree": 2,
                "dp_degree": 2,
                "tp_degree": 1,
            }, ("normal PP=2/DP=2/TP=1 AutoEP checkpoint topology was misclassified: "
                f"{source_checkpoint['discovered_topology']}")

            universal_dir = _convert_checkpoint(save_dir, tag)
            universal_summary = _rank_zero_call(lambda: _inspect_universal_checkpoint(universal_dir, expected))
            if dist.get_rank() == 0:
                print("AUTOEP_PP2_UNIVERSAL=" + json.dumps(universal_summary, sort_keys=True))
            assert not universal_summary["problems"], universal_summary["problems"]

            engine.destroy()
            engine = None
            seed_everything(8198)
            reloaded = _make_pipeline_engine(load_universal=True,
                                             use_data_before_expert_parallelism=use_data_before_expert_parallelism)
            loaded_runtime = _runtime_metadata(reloaded)
            _assert_runtime_metadata(loaded_runtime)
            load_path, _ = reloaded.load_checkpoint(save_dir, tag=f"{tag}_universal")
            assert load_path is not None
            loaded, loaded_coverage = _snapshot_expected_expert_states(reloaded, num_experts=4)
            _assert_snapshots_equal(loaded, expected)
            assert loaded_coverage == coverage
            _assert_expert_model_weights(reloaded, expected)
            _assert_nested_exact(_local_optimizer_state(reloaded), expected_optimizer, "optimizer")

            actual_nonexpert = _nonexpert_model_state(reloaded)
            assert actual_nonexpert.keys() == expected_nonexpert.keys()
            for param_name, expected_param in expected_nonexpert.items():
                torch.testing.assert_close(actual_nonexpert[param_name], expected_param, rtol=0, atol=0)
            pre_step = _optimizer_steps(reloaded)
            post_load_losses = _train_batches(reloaded, count=1, seed=8200)
            post_step = _optimizer_steps(reloaded)
            assert post_step == [step + 1 for step in pre_step]
            if dist.get_rank() == 0:
                print("AUTOEP_PP2_LOAD=" + json.dumps(
                    {
                        "runtime": loaded_runtime,
                        "losses": post_load_losses,
                        "expert_state_exact": True,
                        "nonexpert_model_exact": True,
                        "optimizer_state_exact": True,
                        "optimizer_step_before": pre_step,
                        "optimizer_step_after": post_step,
                    },
                    sort_keys=True))
        finally:
            if reloaded is not None:
                reloaded.destroy()
            if engine is not None:
                engine.destroy()
