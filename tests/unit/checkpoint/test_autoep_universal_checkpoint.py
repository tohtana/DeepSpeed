# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP Universal Checkpoint coverage for ZeRO stages 1 and 2."""

import os
from types import SimpleNamespace

import deepspeed
import deepspeed.comm as dist
import pytest
import torch

from deepspeed.checkpoint.constants import (
    AUTOEP_LAYERS_KEY,
    EP_IS_EXPERT_PARAM,
    EP_NUM_EXPERTS,
    PARAM,
    PARAM_SHAPES,
)
from deepspeed.checkpoint.autoep_universal import _autoep_zero12_dp_ranks
from deepspeed.checkpoint.ds_to_universal import _aggregate_autoep_zero12_metadata, main as convert_to_universal
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.utils.groups import _get_expert_parallel_ranks
from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import (
    MockMoETransformer,
    engine_input_dtype,
    make_autoep_config,
    seed_everything,
    skip_unless_h100_tests_enabled,
)

EXPERT_STATE_KEYS = ("fp32", "exp_avg", "exp_avg_sq")


def _metadata_entry(module_path):
    return {
        "moe_layer_id": 0,
        "module_path": module_path,
        "num_experts": 4,
        "num_local_experts": 2,
        "ep_size": 2,
        "expert_key_prefix": f"{module_path}.experts",
    }


def _write_model_state(path, metadata, use_data_before_expert_parallelism=False):
    param_shapes = [{f"{entry['module_path']}.weight": torch.Size([1]) for entry in metadata}]
    torch.save(
        {
            AUTOEP_LAYERS_KEY: metadata,
            PARAM_SHAPES: param_shapes,
            "ds_config": {
                "use_data_before_expert_parallelism": use_data_before_expert_parallelism,
            },
        },
        path,
    )


def test_aggregate_autoep_zero12_metadata_unions_pipeline_stages(tmpdir):
    stage0_path = os.path.join(str(tmpdir), "mp_rank_00_model_states.pt")
    stage1_path = os.path.join(str(tmpdir), "mp_rank_01_model_states.pt")
    _write_model_state(stage0_path, [_metadata_entry("0.mlp")])
    _write_model_state(stage1_path, [_metadata_entry("2.mlp")])

    param_shapes, metadata, use_data_before_expert_parallelism = _aggregate_autoep_zero12_metadata(
        [stage0_path, stage1_path])

    assert param_shapes == [{"0.mlp.weight": torch.Size([1])}, {"2.mlp.weight": torch.Size([1])}]
    assert [entry["module_path"] for entry in metadata] == ["0.mlp", "2.mlp"]
    assert use_data_before_expert_parallelism is False


def test_aggregate_autoep_zero12_metadata_rejects_conflicting_prefix(tmpdir):
    stage0_path = os.path.join(str(tmpdir), "mp_rank_00_model_states.pt")
    stage1_path = os.path.join(str(tmpdir), "mp_rank_01_model_states.pt")
    stage0 = _metadata_entry("0.mlp")
    stage1 = dict(stage0, num_experts=8)
    _write_model_state(stage0_path, [stage0])
    _write_model_state(stage1_path, [stage1])

    with pytest.raises(RuntimeError, match="Conflicting AutoEP metadata"):
        _aggregate_autoep_zero12_metadata([stage0_path, stage1_path])


def test_aggregate_autoep_zero12_metadata_rejects_ordering_disagreement(tmpdir):
    stage0_path = os.path.join(str(tmpdir), "mp_rank_00_model_states.pt")
    stage1_path = os.path.join(str(tmpdir), "mp_rank_01_model_states.pt")
    _write_model_state(stage0_path, [_metadata_entry("0.mlp")])
    _write_model_state(stage1_path, [_metadata_entry("2.mlp")], use_data_before_expert_parallelism=True)

    with pytest.raises(RuntimeError, match="disagrees across model-state files"):
        _aggregate_autoep_zero12_metadata([stage0_path, stage1_path])


@pytest.mark.parametrize("use_data_before_expert_parallelism", [False, True], ids=["E+D", "D+E"])
def test_autoep_zero12_dp_rank_mapping_matches_pipeline_runtime_groups(use_data_before_expert_parallelism):
    for world_size in range(2, 65):
        for pp_degree in (1, 2, 4):
            if world_size % pp_degree != 0:
                continue
            dp_degree = world_size // pp_degree
            for ep_size in range(1, dp_degree + 1):
                if dp_degree % ep_size != 0:
                    continue
                ep_groups, edp_groups = _get_expert_parallel_ranks(
                    world_size=world_size,
                    tensor_parallel_size_=1,
                    expert_parallel_size_=ep_size,
                    pipeline_parallel_size_=pp_degree,
                    use_data_before_expert_parallel_=use_data_before_expert_parallelism)

                for stage_start in range(0, world_size, dp_degree):
                    stage_end = stage_start + dp_degree
                    ep_rank_by_global_rank = {}
                    for group in ep_groups:
                        if not all(stage_start <= rank < stage_end for rank in group):
                            continue
                        for ep_rank, global_rank in enumerate(group):
                            ep_rank_by_global_rank[global_rank] = ep_rank

                    runtime_classes = []
                    for ep_rank in range(ep_size):
                        runtime_local_ranks = sorted(global_rank - stage_start
                                                     for global_rank, rank_ep in ep_rank_by_global_rank.items()
                                                     if rank_ep == ep_rank)
                        conversion_local_ranks = list(
                            _autoep_zero12_dp_ranks(ep_rank, dp_degree, ep_size, use_data_before_expert_parallelism))
                        assert conversion_local_ranks == runtime_local_ranks
                        runtime_classes.append(runtime_local_ranks)

                    stage_edp_groups = sorted(
                        sorted(global_rank - stage_start for global_rank in group) for group in edp_groups
                        if all(stage_start <= rank < stage_end for rank in group))
                    assert sorted(runtime_classes) == stage_edp_groups


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
        raise RuntimeError(f"AutoEP Universal Checkpoint conversion failed: {conversion_error[0]}")
    return universal_dir


def _expert_params(engine):
    for module_name, module in engine.module.named_modules():
        if not isinstance(module, AutoEPMoELayer):
            continue
        prefix = f"{module_name}." if module_name else ""
        for weight_name in ("w1", "w2", "w3"):
            yield f"{prefix}experts.{weight_name}", module, getattr(module.experts, weight_name)


def _nonexpert_model_state(engine):
    expert_names = {param_name for param_name, _, _ in _expert_params(engine)}
    return {
        param_name: param.detach().cpu().clone()
        for param_name, param in engine.module.named_parameters() if param_name not in expert_names
    }


def _run_steps(engine, count):
    for _ in range(count):
        inputs = torch.randn(1, 8, 64, device=engine.device, dtype=engine_input_dtype(engine))
        loss = engine(inputs).float().mean()
        engine.backward(loss)
        engine.step()
        assert torch.isfinite(loss)


def _load_expert_state(universal_dir, param_name, state_key, num_experts):
    state = torch.load(os.path.join(universal_dir, "zero", param_name, f"{state_key}.pt"),
                       map_location="cpu",
                       weights_only=False)
    assert state[EP_IS_EXPERT_PARAM] is True
    assert state[EP_NUM_EXPERTS] == num_experts
    assert state[PARAM].shape[0] == num_experts
    assert state[PARAM].dtype == torch.float32
    return state[PARAM]


def _collect_expected_expert_states(engine, num_experts):
    local_records = []
    local_errors = []
    for param_name, module, param in _expert_params(engine):
        mapping = getattr(param, "_hp_mapping", None)
        if mapping is None:
            continue

        address = mapping.lp_fragment_address
        record = {
            "param_name": param_name,
            "ep_rank": module.ep_rank,
            "ep_size": module.ep_size,
            "local_shape": tuple(param.shape),
            "start": address.start,
            "numel": address.numel,
            "states": {},
        }
        for state_key in EXPERT_STATE_KEYS:
            flat_param = engine.optimizer.single_partition_of_fp32_groups[mapping.param_group_index]
            hp_address = mapping.hp_fragment_address
            if state_key == "fp32":
                fragment = flat_param.narrow(0, hp_address.start, hp_address.numel)
            else:
                optimizer_state = engine.optimizer.optimizer.state[flat_param]
                if state_key not in optimizer_state:
                    local_errors.append(f"{param_name}/{state_key}: missing from the underlying optimizer state")
                    continue
                fragment = optimizer_state[state_key].narrow(0, hp_address.start, hp_address.numel)
            record["states"][state_key] = fragment.detach().float().cpu().clone()
        local_records.append(record)

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, {"records": local_records, "errors": local_errors})
    errors = [error for payload in gathered for error in payload["errors"]]
    assert not errors, f"failed to snapshot source expert optimizer fragments: {errors}"

    expected = {}
    records = [record for payload in gathered for record in payload["records"]]
    param_names = sorted({record["param_name"] for record in records})
    for param_name in param_names:
        param_records = [record for record in records if record["param_name"] == param_name]
        ep_sizes = {record["ep_size"] for record in param_records}
        local_shapes = {record["local_shape"] for record in param_records}
        assert len(ep_sizes) == 1
        assert len(local_shapes) == 1
        ep_size = ep_sizes.pop()
        local_shape = local_shapes.pop()
        assert local_shape[0] * ep_size == num_experts

        expected[param_name] = {}
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
                ep_tensors.append(tensor.reshape(local_shape))
            expected[param_name][state_key] = torch.cat(ep_tensors, dim=0)
    return expected


def _assert_universal_matches_expected(universal_dir, expected, num_experts):
    nonzero_moments = [False, False]
    for param_name, state_by_key in expected.items():
        for state_key, expected_state in state_by_key.items():
            actual = _load_expert_state(universal_dir, param_name, state_key, num_experts)
            torch.testing.assert_close(actual, expected_state, rtol=0, atol=0)
            if state_key != "fp32" and torch.count_nonzero(actual).item() > 0:
                nonzero_moments[EXPERT_STATE_KEYS.index(state_key) - 1] = True
    assert nonzero_moments == [True, True]


def _assert_model_weights_match_universal(engine, universal_dir, num_experts):
    mismatches = []
    for param_name, module, param in _expert_params(engine):
        local_experts = num_experts // module.ep_size
        start = module.ep_rank * local_experts
        end = start + local_experts
        full_state = _load_expert_state(universal_dir, param_name, "fp32", num_experts)
        expected = full_state[start:end].to(device=param.device, dtype=param.dtype)
        if not torch.equal(param.detach(), expected):
            max_diff = (param.detach().float() - expected.float()).abs().max().item()
            mismatches.append(f"{param_name}: loaded model weight differs (max abs diff {max_diff})")

    rank_mismatches = [None] * dist.get_world_size()
    dist.all_gather_object(rank_mismatches, mismatches)
    assert not any(rank_mismatches), f"loaded expert model-weight mismatches by rank: {rank_mismatches}"


class TestAutoEPZero12UniversalCheckpoint(DistributedTest):
    world_size = 4

    @pytest.mark.parametrize("zero_stage", [1, 2])
    @pytest.mark.parametrize("use_data_before_expert_parallelism", [False, True], ids=["E+D", "D+E"])
    def test_save_convert_load_preserves_expert_optimizer_state(self, tmpdir, zero_stage,
                                                                use_data_before_expert_parallelism):
        skip_unless_h100_tests_enabled("AutoEP ZeRO-1/2 Universal Checkpoint coverage requires H100")
        seed_everything(8147 + zero_stage + int(use_data_before_expert_parallelism))
        num_experts = 4
        ep_size = 2
        config = make_autoep_config(zero_stage=zero_stage, ep_size=ep_size, mixed_precision=True)
        config["use_data_before_expert_parallelism"] = use_data_before_expert_parallelism
        engine, _, _, _ = deepspeed.initialize(model=MockMoETransformer(num_experts=num_experts), config=config)
        _run_steps(engine, 2)
        expected = _collect_expected_expert_states(engine, num_experts)
        expected_nonexpert = _nonexpert_model_state(engine)

        save_dir = str(tmpdir)
        tag = f"autoep-zero{zero_stage}"
        engine.save_checkpoint(save_dir, tag=tag)
        universal_dir = _convert_checkpoint(save_dir, tag)
        _assert_universal_matches_expected(universal_dir, expected, num_experts)
        engine.destroy()

        load_config = make_autoep_config(zero_stage=zero_stage, ep_size=4, mixed_precision=True)
        load_config["use_data_before_expert_parallelism"] = use_data_before_expert_parallelism
        load_config["checkpoint"] = {"load_universal": True}
        reloaded, _, _, _ = deepspeed.initialize(model=MockMoETransformer(num_experts=num_experts), config=load_config)
        reloaded.load_checkpoint(save_dir, tag=f"{tag}_universal")
        loaded = _collect_expected_expert_states(reloaded, num_experts)
        _assert_universal_matches_expected(universal_dir, loaded, num_experts)
        _assert_model_weights_match_universal(reloaded, universal_dir, num_experts)
        actual_nonexpert = _nonexpert_model_state(reloaded)
        assert actual_nonexpert.keys() == expected_nonexpert.keys()
        for param_name, expected_param in expected_nonexpert.items():
            torch.testing.assert_close(actual_nonexpert[param_name], expected_param, rtol=0, atol=0)
        _run_steps(reloaded, 1)
        reloaded.destroy()
