# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import copy
from datetime import timedelta
import importlib
import json
import multiprocessing
import operator
from pathlib import Path
import queue
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import pytest
import torch

import deepspeed.compile.agent_runner as agent_runner_module
import deepspeed.compile.backend as backend
import deepspeed.compile.optimizer as optimizer_module
import deepspeed.comm as dist
from deepspeed.compile.agent_runner import AgentRunner, AgentRunnerConfig
from deepspeed.compile.config import CompileConfig
from deepspeed.compile.graph_edit import (GraphEditPayload, RUNTIME_GRAPH_ID, finalize_graph_edit,
                                          structural_fingerprint)
from deepspeed.compile.optimizer import (OptimizationContext, OptimizationResult, TwoAgentLoopOptimizer)
from deepspeed.compile.profilers import ProfilingResult


init_z3_module = importlib.import_module("deepspeed.compile.init_z3")
_DISTRIBUTED_MEMORY_PROFILE_ENTERED = False


class ChainModule(torch.nn.Module):

    def forward(self, x):
        return torch.sigmoid(torch.relu(x))


class StatefulModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.register_buffer("running", torch.tensor(3.0))

    def forward(self, x):
        return torch.sigmoid(torch.relu(x * self.weight + self.running))


class TwoInputModule(torch.nn.Module):

    def forward(self, value, unused):
        return torch.relu(value)


class _GlooTimingProfiler:

    def __init__(self, gm, debug_log=False):
        self.gm = gm

    def run(self, *args):
        return self.gm(*args)


class _GlooMemoryProfiler:

    def __init__(self, gm, debug_log=False):
        self.gm = gm
        self.profile_complete = True
        self.mem_record = []

    def run(self, *args):
        global _DISTRIBUTED_MEMORY_PROFILE_ENTERED
        _DISTRIBUTED_MEMORY_PROFILE_ENTERED = True
        phase_token = torch.tensor([1])
        dist.all_reduce(phase_token)
        output = self.gm(*args)
        self.mem_record = [(node.name, 1, 0, 1) for node in self.gm.graph.nodes]
        return output


class StaticRunner:

    def __init__(self, callback, command):
        self.callback = callback
        self.config = SimpleNamespace(command=[command])
        self.calls = []

    def run(self, prompt, iteration_dir, role=None, artifact_prefix=None):
        payload = json.loads(prompt)
        response = self.callback(payload, role, len(self.calls))
        self.calls.append((role, payload))
        prefix = artifact_prefix if artifact_prefix is not None else role
        prefix = f"{prefix}_" if prefix else ""
        iteration_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = iteration_dir / f"{prefix}prompt.txt"
        stdout_path = iteration_dir / f"{prefix}stdout.txt"
        stderr_path = iteration_dir / f"{prefix}stderr.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        stdout_path.write_text(json.dumps(response), encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return SimpleNamespace(stdout=json.dumps(response),
                               stderr="",
                               returncode=0,
                               timed_out=False,
                               prompt_path=str(prompt_path),
                               stdout_path=str(stdout_path),
                               stderr_path=str(stderr_path))


def _config(**kwargs):
    values = {
        "zero3_tuning_strategy": "agent",
        "agent_architecture": "two_agent",
        "agent_evaluator_command": ["evaluate"],
        "agent_optimizer_command": ["optimize"],
        "agent_max_iterations": 1,
        "agent_max_retries_per_iteration": 1,
        "agent_timeout_sec": 5,
    }
    values.update(kwargs)
    return CompileConfig(**values)


def _context_from_gm(config, gm, create_inputs_fn):
    names = [node.name for node in gm.graph.nodes]
    profile = ProfilingResult(fwd_graph=gm.graph,
                              fwd_mem=[(name, 1, 0, 1) for name in names],
                              fwd_time=[(name, 1.0, 1.0) for name in names],
                              fwd_tensor_sizes=[(name, 4) for name in names],
                              fwd_mem_complete=True,
                              needs_backward=False)
    ctx = OptimizationContext(gm=gm,
                              graph_id=11,
                              graph_slot=(0, "fwd"),
                              graph_order=[(11, False)],
                              profiling_results={11: profile},
                              create_inputs_fn=create_inputs_fn,
                              bwd=False,
                              debug_log=False,
                              compile_config=config)
    return gm, ctx


def _context(config, module=None):
    gm = torch.fx.symbolic_trace(module or ChainModule())
    return _context_from_gm(config, gm, lambda: (torch.tensor([-2.0, 3.0]),))


def _aot_lifted_context(config):
    graph = torch.fx.Graph()
    state = graph.placeholder("state")
    weights = graph.call_function(operator.getitem, (state, "weights"))
    weight = graph.call_function(operator.getitem, (weights, 0))
    value = graph.placeholder("value")
    result_node = graph.call_function(operator.mul, (value, weight))
    graph.output(result_node)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    lifted_parameter = torch.nn.Parameter(torch.tensor(2.0))

    def create_inputs():
        return ({"weights": [lifted_parameter]}, torch.tensor(3.0))

    _, ctx = _context_from_gm(config, gm, create_inputs)
    return gm, ctx, lifted_parameter


def _rank_divergent_output_abi_worker(rank, init_method, result_queue):
    global _DISTRIBUTED_MEMORY_PROFILE_ENTERED
    _DISTRIBUTED_MEMORY_PROFILE_ENTERED = False
    try:
        dist.init_distributed(dist_backend="gloo",
                              auto_mpi_discovery=False,
                              init_method=init_method,
                              rank=rank,
                              world_size=2,
                              timeout=timedelta(seconds=10),
                              verbose=False)

        def rank_local_output(value):
            return value if rank == 0 else [value]

        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        local_result = graph.call_function(rank_local_output, (value, ))
        output = graph.output(value)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        _, ctx = _context_from_gm(_config(), gm, lambda: (torch.tensor([1.0]), ))
        accepted_calls = optimizer_module._capture_profile_calls(ctx)
        accepted_output = gm(*accepted_calls[0].args)
        ctx.runtime_abi = optimizer_module._runtime_abi_descriptor(gm, accepted_calls, accepted_output)

        raw = GraphEditPayload(generation=1,
                               graph_slot=ctx.graph_slot,
                               base_fingerprint=structural_fingerprint(gm, ctx.graph_id),
                               expected_result_fingerprint=None,
                               operations=[{
                                   "op": "set_args_kwargs",
                                   "id": "base:2",
                                   "args": [{"node": "base:1"}],
                               }, {
                                   "op": "reorder",
                                   "order": ["base:0", "base:1", "base:2"],
                               }],
                               reason="Exercise rank-divergent local output ABI synchronization")
        payload, _ = finalize_graph_edit(gm, raw, ctx.graph_id)
        optimizer_module.ProfilingInterpreter = _GlooTimingProfiler
        optimizer_module.MemoryProfilingInterpreter = _GlooMemoryProfiler
        optimizer_module.is_profile_incomplete = lambda _graph: False

        _, candidate_profile, records = TwoAgentLoopOptimizer._apply_and_profile_candidate(ctx, payload)
        result_queue.put({
            "rank": rank,
            "success": candidate_profile is None and all(not record["success"] for record in records),
            "abi_incompatible": all(record["abi_compatible"] is False for record in records),
            "memory_entered": _DISTRIBUTED_MEMORY_PROFILE_ENTERED,
            "error": None,
        })
    except Exception:
        result_queue.put({
            "rank": rank,
            "success": False,
            "abi_incompatible": False,
            "memory_entered": _DISTRIBUTED_MEMORY_PROFILE_ENTERED,
            "error": traceback.format_exc(),
        })
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _fake_profile(gm, ctx, profile_calls=None, expected_abi=None):
    if profile_calls is None:
        profile_calls = optimizer_module._capture_profile_calls(ctx)
    optimizer_module._validate_runtime_call_contract(gm, profile_calls)
    output = gm(*profile_calls[0].args, **profile_calls[0].kwargs)
    runtime_abi = optimizer_module._runtime_abi_descriptor(gm, profile_calls, output)
    if expected_abi is not None and runtime_abi != expected_abi:
        raise optimizer_module._RuntimeABIError("Candidate output ABI differs from the accepted graph")
    profile = copy.deepcopy(ctx.profiling_results[ctx.graph_id])
    names = [node.name for node in gm.graph.nodes]
    for node in gm.graph.nodes:
        node.meta.update({"device_time": 1.0, "wall_time": 1.0, "tensor_size": 4})
    profile.fwd_graph = gm.graph
    profile.fwd_mem = [(name, 1, 0, 1) for name in names]
    profile.fwd_time = [(name, 1.0, 1.0) for name in names]
    profile.fwd_tensor_sizes = [(name, 4) for name in names]
    profile.fwd_mem_complete = True
    return profile, runtime_abi


def test_profile_graph_single_rank_runs_timing_then_memory(monkeypatch):
    config = _config()
    gm, ctx = _context(config)
    phases = []

    class TimingProfiler:

        def __init__(self, profile_gm, debug_log=False):
            self.gm = profile_gm

        def run(self, *args):
            phases.append("timing")
            return self.gm(*args)

    class MemoryProfiler:

        def __init__(self, profile_gm, debug_log=False):
            self.gm = profile_gm
            self.profile_complete = True
            self.mem_record = []

        def run(self, *args):
            phases.append("memory")
            output = self.gm(*args)
            self.mem_record = [(node.name, 1, 0, 1) for node in self.gm.graph.nodes]
            return output

    monkeypatch.setattr(optimizer_module, "ProfilingInterpreter", TimingProfiler)
    monkeypatch.setattr(optimizer_module, "MemoryProfilingInterpreter", MemoryProfiler)
    monkeypatch.setattr(optimizer_module, "is_profile_incomplete", lambda _graph: False)

    profile, runtime_abi = optimizer_module._profile_graph(gm, ctx)

    assert phases == ["timing", "memory"]
    assert profile.fwd_mem_complete
    assert runtime_abi.output_leaf_kinds == ("tensor", )


def _metadata_optimizer_response(prompt, role, _call_index):
    assert role == "optimizer"
    snapshot = prompt["graph_snapshot"]
    node_ids = [node["id"] for node in prompt["graph_nodes"]]
    return {
        "schema_version": 1,
        "generation": snapshot["generation"] + 1,
        "graph_slot": [snapshot["graph_slot"]["index"], snapshot["graph_slot"]["direction"]],
        "base_fingerprint": snapshot["graph_fingerprint"],
        "expected_result_fingerprint": None,
        "reason": "Probe rollback of profiling state",
        "operations": [{
            "op": "patch_meta",
            "id": node_ids[0],
            "meta": {"candidate_state_probe": True},
        }, {
            "op": "reorder",
            "order": node_ids,
        }],
    }


def _evaluator_response(accept_candidate):

    def callback(prompt, role, _call_index):
        assert role == "evaluator"
        snapshot = prompt["accepted_graph"]["snapshot"]
        if prompt["stage"] == "accepted_graph":
            return {
                "schema_version": 3,
                "based_on": snapshot,
                "decision": "continue",
                "summary": "Try a generic replacement",
                "optimizer_brief": "Replace sigmoid with any valid FX topology",
                "candidate_generation": None,
                "candidate_fingerprint": None,
            }
        return {
            "schema_version": 3,
            "based_on": snapshot,
            "decision": "accept" if accept_candidate else "reject",
            "summary": "Candidate decision",
            "optimizer_brief": "",
            "candidate_generation": prompt["candidate"]["generation"],
            "candidate_fingerprint": prompt["candidate"]["fingerprint"],
        }

    return callback


def _optimizer_response(prompt, role, _call_index):
    assert role == "optimizer"
    snapshot = prompt["graph_snapshot"]
    return {
        "schema_version": 1,
        "generation": snapshot["generation"] + 1,
        "graph_slot": [snapshot["graph_slot"]["index"], snapshot["graph_slot"]["direction"]],
        "base_fingerprint": snapshot["graph_fingerprint"],
        "expected_result_fingerprint": None,
        "reason": "Replace sigmoid with negation",
        "operations": [{
            "op": "create_node",
            "id": "new:neg",
            "node_op": "call_function",
            "target": "torch.neg",
            "args": [{"node": "base:1"}],
            "kwargs": {},
            "copy_meta_from": "base:2",
        }, {
            "op": "set_args_kwargs",
            "id": "base:3",
            "args": [{"node": "new:neg"}],
        }, {
            "op": "delete_node",
            "id": "base:2",
        }, {
            "op": "reorder",
            "order": ["base:0", "base:1", "new:neg", "base:3"],
        }],
    }


def _delete_placeholder_response(prompt, role, _call_index):
    assert role == "optimizer"
    snapshot = prompt["graph_snapshot"]
    placeholder_ids = [node["id"] for node in prompt["graph_nodes"] if node["op"] == "placeholder"]
    deleted_id = placeholder_ids[-1]
    remaining_ids = [node["id"] for node in prompt["graph_nodes"] if node["id"] != deleted_id]
    return {
        "schema_version": 1,
        "generation": snapshot["generation"] + 1,
        "graph_slot": [snapshot["graph_slot"]["index"], snapshot["graph_slot"]["direction"]],
        "base_fingerprint": snapshot["graph_fingerprint"],
        "expected_result_fingerprint": None,
        "reason": "Delete an unused placeholder",
        "operations": [{
            "op": "delete_node",
            "id": deleted_id,
        }, {
            "op": "reorder",
            "order": remaining_ids,
        }],
    }


def _change_output_structure_response(prompt, role, _call_index):
    assert role == "optimizer"
    snapshot = prompt["graph_snapshot"]
    node_ids = [node["id"] for node in prompt["graph_nodes"]]
    output_id = next(node["id"] for node in prompt["graph_nodes"] if node["op"] == "output")
    output_value = prompt["graph_nodes"][-1]["args"][0]
    return {
        "schema_version": 1,
        "generation": snapshot["generation"] + 1,
        "graph_slot": [snapshot["graph_slot"]["index"], snapshot["graph_slot"]["direction"]],
        "base_fingerprint": snapshot["graph_fingerprint"],
        "expected_result_fingerprint": None,
        "reason": "Wrap the existing output in a tuple",
        "operations": [{
            "op": "set_args_kwargs",
            "id": output_id,
            "args": [{"tuple": [output_value]}],
        }, {
            "op": "reorder",
            "order": node_ids,
        }],
    }


def _rename_placeholder_response(prompt, role, _call_index):
    assert role == "optimizer"
    snapshot = prompt["graph_snapshot"]
    return {
        "schema_version": 1,
        "generation": snapshot["generation"] + 1,
        "graph_slot": [snapshot["graph_slot"]["index"], snapshot["graph_slot"]["direction"]],
        "base_fingerprint": snapshot["graph_fingerprint"],
        "expected_result_fingerprint": None,
        "reason": "Replace the positional placeholder with a differently named one",
        "operations": [{
            "op": "create_node",
            "id": "new:renamed_input",
            "node_op": "placeholder",
            "target": "renamed_input",
            "args": [],
            "kwargs": {},
        }, {
            "op": "set_args_kwargs",
            "id": "base:1",
            "args": [{"node": "new:renamed_input"}],
        }, {
            "op": "delete_node",
            "id": "base:0",
        }, {
            "op": "reorder",
            "order": ["new:renamed_input", "base:1", "base:2", "base:3"],
        }],
    }


def test_two_agent_loop_profiles_broadcasts_and_accepts_generic_edit(monkeypatch):
    config = _config()
    gm, ctx = _context(config)
    evaluator = StaticRunner(_evaluator_response(True), "evaluate")
    graph_optimizer = StaticRunner(_optimizer_response, "optimize")
    monkeypatch.setattr(optimizer_module, "_profile_graph", _fake_profile)
    monkeypatch.setattr(optimizer_module, "_cleanup_after_candidate", lambda: None)

    result = TwoAgentLoopOptimizer(evaluator, graph_optimizer, config).optimize(gm, ctx)

    assert torch.equal(gm(torch.tensor([-2.0, 3.0])), torch.tensor([0.0, -3.0]))
    assert [entry.action for entry in result.trace] == ["evaluate", "accepted"]
    assert [role for role, _ in evaluator.calls] == ["evaluator", "evaluator"]
    assert [role for role, _ in graph_optimizer.calls] == ["optimizer"]
    optimizer_contract = graph_optimizer.calls[0][1]["edit_contract"]
    assert "registered operator" not in json.dumps(optimizer_contract).lower()
    assert any("There are no semantic operator allowlists or bans" in rule
               for rule in optimizer_contract["mechanical_rules"])
    assert graph_optimizer.calls[0][1]["graph_runtime"]["graph_id"] == RUNTIME_GRAPH_ID
    assert evaluator.calls[0][1]["graph_runtime"]["graph_id"] == RUNTIME_GRAPH_ID


def test_rejected_candidate_leaves_accepted_graph_intact(monkeypatch):
    config = _config()
    gm, ctx = _context(config)
    original_fingerprint = structural_fingerprint(gm)
    evaluator = StaticRunner(_evaluator_response(False), "evaluate")
    graph_optimizer = StaticRunner(_optimizer_response, "optimize")
    monkeypatch.setattr(optimizer_module, "_profile_graph", _fake_profile)
    monkeypatch.setattr(optimizer_module, "_cleanup_after_candidate", lambda: None)

    result = TwoAgentLoopOptimizer(evaluator, graph_optimizer, config).optimize(gm, ctx)

    assert structural_fingerprint(gm) == original_fingerprint
    assert any(node.target == torch.sigmoid for node in gm.graph.nodes)
    assert result.trace[-1].action == "rejected"


def test_placeholder_removal_is_rejected_before_candidate_evaluator(monkeypatch):
    config = _config()
    gm = torch.fx.symbolic_trace(TwoInputModule())
    gm, ctx = _context_from_gm(config, gm, lambda: (torch.tensor([-1.0, 2.0]), torch.tensor(0.0)))
    evaluator = StaticRunner(_evaluator_response(True), "evaluate")
    graph_optimizer = StaticRunner(_delete_placeholder_response, "optimize")
    monkeypatch.setattr(optimizer_module, "_profile_graph", _fake_profile)
    monkeypatch.setattr(optimizer_module, "_cleanup_after_candidate", lambda: None)

    result = TwoAgentLoopOptimizer(evaluator, graph_optimizer, config).optimize(gm, ctx)

    assert result.trace[-1].action == "rejected"
    assert "runtime ABI" in result.trace[-1].summary
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0][1]["stage"] == "accepted_graph"
    assert len([node for node in gm.graph.nodes if node.op == "placeholder"]) == 2


def test_output_structure_change_is_rejected_before_candidate_evaluator(monkeypatch):
    config = _config()
    gm, ctx = _context(config)
    evaluator = StaticRunner(_evaluator_response(True), "evaluate")
    graph_optimizer = StaticRunner(_change_output_structure_response, "optimize")
    monkeypatch.setattr(optimizer_module, "_profile_graph", _fake_profile)
    monkeypatch.setattr(optimizer_module, "_cleanup_after_candidate", lambda: None)

    result = TwoAgentLoopOptimizer(evaluator, graph_optimizer, config).optimize(gm, ctx)

    assert result.trace[-1].action == "rejected"
    assert "runtime ABI" in result.trace[-1].summary
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0][1]["stage"] == "accepted_graph"
    assert isinstance(gm(torch.tensor([-1.0, 2.0])), torch.Tensor)


def test_rank_divergent_output_abi_stops_before_memory_collectives(tmp_path):
    process_context = multiprocessing.get_context("spawn")
    result_queue = process_context.Queue()
    init_method = f"file://{tmp_path / 'abi-phase-store'}"
    processes = [
        process_context.Process(target=_rank_divergent_output_abi_worker, args=(rank, init_method, result_queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + 20
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))
    alive = [process for process in processes if process.is_alive()]
    if alive:
        for process in alive:
            process.terminate()
        for process in alive:
            process.join(5)
        pytest.fail(f"rank-divergent ABI workers hung: {[process.pid for process in alive]}")

    results = []
    try:
        for _ in processes:
            results.append(result_queue.get(timeout=2))
    except queue.Empty:
        pytest.fail(f"distributed ABI workers exited without results: {[process.exitcode for process in processes]}")

    assert all(process.exitcode == 0 for process in processes)
    assert {result["rank"] for result in results} == {0, 1}
    assert all(result["error"] is None for result in results), results
    assert all(result["success"] for result in results)
    assert all(result["abi_incompatible"] for result in results)
    assert all(not result["memory_entered"] for result in results)


def test_placeholder_name_change_with_same_positional_abi_can_be_accepted(monkeypatch):
    config = _config()
    gm, ctx = _context(config)
    evaluator = StaticRunner(_evaluator_response(True), "evaluate")
    graph_optimizer = StaticRunner(_rename_placeholder_response, "optimize")
    monkeypatch.setattr(optimizer_module, "_profile_graph", _fake_profile)
    monkeypatch.setattr(optimizer_module, "_cleanup_after_candidate", lambda: None)

    result = TwoAgentLoopOptimizer(evaluator, graph_optimizer, config).optimize(gm, ctx)

    assert result.trace[-1].action == "accepted"
    placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
    assert placeholder.target == "renamed_input"
    assert torch.equal(gm(torch.tensor([-1.0, 2.0])), torch.sigmoid(torch.relu(torch.tensor([-1.0, 2.0]))))


def test_rejected_candidate_restores_registered_parameters_and_buffers(monkeypatch):
    config = _config()
    gm, ctx = _context(config, StatefulModule())
    original_parameter = next(gm.parameters()).detach().clone()
    original_buffer = next(gm.buffers()).detach().clone()

    def mutating_profile(profile_gm, profile_ctx, profile_calls=None, expected_abi=None):
        if any(node.meta.get("candidate_state_probe") for node in profile_gm.graph.nodes):
            with torch.no_grad():
                next(profile_gm.parameters()).add_(5)
                next(profile_gm.buffers()).add_(7)
        return _fake_profile(profile_gm, profile_ctx, profile_calls, expected_abi)

    evaluator = StaticRunner(_evaluator_response(False), "evaluate")
    graph_optimizer = StaticRunner(_metadata_optimizer_response, "optimize")
    monkeypatch.setattr(optimizer_module, "_profile_graph", mutating_profile)
    monkeypatch.setattr(optimizer_module, "_cleanup_after_candidate", lambda: None)

    result = TwoAgentLoopOptimizer(evaluator, graph_optimizer, config).optimize(gm, ctx)

    assert result.trace[-1].action == "rejected"
    assert torch.equal(next(gm.parameters()), original_parameter)
    assert torch.equal(next(gm.buffers()), original_buffer)


def test_rejected_candidate_restores_nested_aot_lifted_parameter_input(monkeypatch):
    config = _config()
    gm, ctx, lifted_parameter = _aot_lifted_context(config)
    assert not list(gm.parameters())
    assert not list(gm.buffers())

    def mutating_profile(profile_gm, profile_ctx, profile_calls=None, expected_abi=None):
        if any(node.meta.get("candidate_state_probe") for node in profile_gm.graph.nodes):
            assert profile_calls is not None
            captured_parameter = profile_calls[0].args[0]["weights"][0]
            assert captured_parameter is lifted_parameter
            with torch.no_grad():
                captured_parameter.add_(10)
        return _fake_profile(profile_gm, profile_ctx, profile_calls, expected_abi)

    evaluator = StaticRunner(_evaluator_response(False), "evaluate")
    graph_optimizer = StaticRunner(_metadata_optimizer_response, "optimize")
    monkeypatch.setattr(optimizer_module, "_profile_graph", mutating_profile)
    monkeypatch.setattr(optimizer_module, "_cleanup_after_candidate", lambda: None)

    result = TwoAgentLoopOptimizer(evaluator, graph_optimizer, config).optimize(gm, ctx)

    assert result.trace[-1].action == "rejected"
    assert torch.equal(lifted_parameter, torch.tensor(2.0))


def test_accepted_profile_restores_registered_state_on_success(monkeypatch):
    config = _config()
    gm, ctx = _context(config, StatefulModule())
    original_parameter = next(gm.parameters()).detach().clone()
    original_buffer = next(gm.buffers()).detach().clone()

    def mutating_profile(profile_gm, profile_ctx, profile_calls=None, expected_abi=None):
        with torch.no_grad():
            next(profile_gm.parameters()).add_(5)
            next(profile_gm.buffers()).add_(7)
        return _fake_profile(profile_gm, profile_ctx, profile_calls, expected_abi)

    monkeypatch.setattr(optimizer_module, "_profile_graph", mutating_profile)

    success, records = TwoAgentLoopOptimizer._profile_accepted_graph(ctx)

    assert success
    assert all(record["success"] for record in records)
    assert torch.equal(next(gm.parameters()), original_parameter)
    assert torch.equal(next(gm.buffers()), original_buffer)
    assert ctx.runtime_abi is not None


def test_accepted_profile_restores_lifted_state_after_profiling_failure(monkeypatch):
    config = _config()
    _, ctx, lifted_parameter = _aot_lifted_context(config)

    def failing_profile(_gm, _ctx, profile_calls=None, expected_abi=None):
        with torch.no_grad():
            profile_calls[0].args[0]["weights"][0].add_(9)
        raise RuntimeError("profile failed after mutation")

    monkeypatch.setattr(optimizer_module, "_profile_graph", failing_profile)

    success, records = TwoAgentLoopOptimizer._profile_accepted_graph(ctx)

    assert not success
    assert "profile failed after mutation" in records[0]["error"]
    assert torch.equal(lifted_parameter, torch.tensor(2.0))


def test_accepted_profile_restore_failure_raises_before_any_agent(monkeypatch):
    config = _config()
    gm, ctx, lifted_parameter = _aot_lifted_context(config)
    evaluator = StaticRunner(_evaluator_response(False), "evaluate")
    graph_optimizer = StaticRunner(_metadata_optimizer_response, "optimize")

    def mutating_profile(profile_gm, profile_ctx, profile_calls=None, expected_abi=None):
        with torch.no_grad():
            profile_calls[0].args[0]["weights"][0].add_(4)
        return _fake_profile(profile_gm, profile_ctx, profile_calls, expected_abi)

    monkeypatch.setattr(optimizer_module, "_profile_graph", mutating_profile)
    monkeypatch.setattr(optimizer_module, "_restore_candidate_state",
                        lambda snapshots: (_ for _ in ()).throw(RuntimeError("restore failed")))

    with pytest.raises(RuntimeError, match="Accepted graph state restoration failed"):
        TwoAgentLoopOptimizer(evaluator, graph_optimizer, config).optimize(gm, ctx)

    assert not evaluator.calls
    assert not graph_optimizer.calls
    assert not torch.equal(lifted_parameter, torch.tensor(2.0))


def test_candidate_restore_failure_stops_before_candidate_evaluation(monkeypatch):
    config = _config()
    gm, ctx = _context(config)
    evaluator = StaticRunner(_evaluator_response(False), "evaluate")
    graph_optimizer = StaticRunner(_metadata_optimizer_response, "optimize")
    monkeypatch.setattr(optimizer_module, "_profile_graph", _fake_profile)
    original_restore = optimizer_module._restore_candidate_state
    restore_calls = []

    def fail_candidate_restore(snapshots):
        restore_calls.append(snapshots)
        if len(restore_calls) == 1:
            original_restore(snapshots)
            return
        raise RuntimeError("restore failed")

    monkeypatch.setattr(optimizer_module, "_restore_candidate_state", fail_candidate_restore)

    with pytest.raises(RuntimeError, match="Candidate state restoration failed"):
        TwoAgentLoopOptimizer(evaluator, graph_optimizer, config).optimize(gm, ctx)

    assert len(evaluator.calls) == 1
    assert evaluator.calls[0][1]["stage"] == "accepted_graph"


def test_stale_candidate_fingerprint_cannot_be_accepted(monkeypatch):
    config = _config()
    gm, ctx = _context(config)

    def stale_evaluator(prompt, role, _call_index):
        response = _evaluator_response(True)(prompt, role, _call_index)
        if prompt["stage"] == "candidate":
            response["candidate_fingerprint"] = "stale-fingerprint"
        return response

    evaluator = StaticRunner(stale_evaluator, "evaluate")
    graph_optimizer = StaticRunner(_optimizer_response, "optimize")
    monkeypatch.setattr(optimizer_module, "_profile_graph", _fake_profile)
    monkeypatch.setattr(optimizer_module, "_cleanup_after_candidate", lambda: None)

    result = TwoAgentLoopOptimizer(evaluator, graph_optimizer, config).optimize(gm, ctx)

    assert any(entry.action == "abort" and "candidate_fingerprint" in entry.summary for entry in result.trace)
    assert any(node.target == torch.sigmoid for node in gm.graph.nodes)


def test_stale_metadata_only_candidate_response_cannot_accept_a_different_edit(monkeypatch):
    config = _config(agent_max_iterations=2)
    gm, ctx = _context(config)
    first_candidate_fingerprint = []

    def evaluator_callback(prompt, role, _call_index):
        assert role == "evaluator"
        snapshot = prompt["accepted_graph"]["snapshot"]
        if prompt["stage"] == "accepted_graph":
            return {
                "schema_version": 3,
                "based_on": snapshot,
                "decision": "continue",
                "summary": "Try a metadata-only edit",
                "optimizer_brief": "Patch data-only metadata",
                "candidate_generation": None,
                "candidate_fingerprint": None,
            }
        current_fingerprint = prompt["candidate"]["fingerprint"]
        if not first_candidate_fingerprint:
            first_candidate_fingerprint.append(current_fingerprint)
            decision = "reject"
            response_fingerprint = current_fingerprint
        else:
            assert current_fingerprint != first_candidate_fingerprint[0]
            decision = "accept"
            response_fingerprint = first_candidate_fingerprint[0]
        return {
            "schema_version": 3,
            "based_on": snapshot,
            "decision": decision,
            "summary": "Candidate decision",
            "optimizer_brief": "",
            "candidate_generation": prompt["candidate"]["generation"],
            "candidate_fingerprint": response_fingerprint,
        }

    def optimizer_callback(prompt, role, call_index):
        response = _metadata_optimizer_response(prompt, role, call_index)
        response["operations"][0]["meta"] = {"candidate_state_probe": True, "attempt": [call_index]}
        return response

    evaluator = StaticRunner(evaluator_callback, "evaluate")
    graph_optimizer = StaticRunner(optimizer_callback, "optimize")
    monkeypatch.setattr(optimizer_module, "_profile_graph", _fake_profile)
    monkeypatch.setattr(optimizer_module, "_cleanup_after_candidate", lambda: None)

    result = TwoAgentLoopOptimizer(evaluator, graph_optimizer, config).optimize(gm, ctx)

    assert any(entry.action == "abort" and "candidate_fingerprint" in entry.summary for entry in result.trace)
    assert len(graph_optimizer.calls) == 2
    assert not any("attempt" in node.meta for node in gm.graph.nodes)


def test_nonzero_rank_never_invokes_evaluator_or_optimizer(monkeypatch):
    config = _config()
    gm, ctx = _context(config)

    def unexpected_agent_call(*args, **kwargs):
        raise AssertionError("nonzero rank invoked an agent")

    evaluator = StaticRunner(unexpected_agent_call, "evaluate")
    graph_optimizer = StaticRunner(unexpected_agent_call, "optimize")
    monkeypatch.setattr(optimizer_module, "_rank", lambda: 1)
    monkeypatch.setattr(TwoAgentLoopOptimizer, "_snapshot_consensus", lambda self, tracker: (True, []))
    monkeypatch.setattr(TwoAgentLoopOptimizer, "_profile_accepted_graph", lambda self, profile_ctx: (True, []))
    monkeypatch.setattr(optimizer_module, "broadcast_json_payload",
                        lambda payload: {
                            "continue": False,
                            "error": None
                        })

    result = TwoAgentLoopOptimizer(evaluator, graph_optimizer, config).optimize(gm, ctx)

    assert not result.trace
    assert not evaluator.calls
    assert not graph_optimizer.calls


def test_backend_marker_runs_z3_pass_then_two_agent_loop_and_baseline_stays_direct(monkeypatch):
    config = _config()
    gm, ctx = _context(config)
    structural_pass = lambda *args, **kwargs: None
    pass_calls = []
    agent_calls = []

    def fake_run_opt_passes(passes, *args, **kwargs):
        pass_calls.append(list(passes))

    class FakeTwoAgent:

        def __init__(self, evaluator_runner, optimizer_runner, compile_config):
            agent_calls.append((evaluator_runner.config.command, optimizer_runner.config.command, compile_config))

        def optimize(self, graph_module, optimization_context):
            assert graph_module is gm
            return OptimizationResult()

    monkeypatch.setattr(backend, "run_opt_passes", fake_run_opt_passes)
    monkeypatch.setattr(optimizer_module, "TwoAgentLoopOptimizer", FakeTwoAgent)

    result = backend.run_optimization([structural_pass, backend.agent_optimization_loop], gm, 11, (0, "fwd"),
                                      ctx.graph_order, ctx.profiling_results, ctx.create_inputs_fn, 0.0,
                                      {11: object()}, False, config)
    assert isinstance(result, OptimizationResult)
    assert pass_calls == [[structural_pass]]
    assert agent_calls[0][:2] == (["evaluate"], ["optimize"])

    pass_calls.clear()
    assert backend.run_optimization([structural_pass], gm, 11, (0, "fwd"), ctx.graph_order,
                                    ctx.profiling_results, ctx.create_inputs_fn, 0.0, {11: object()}, False,
                                    CompileConfig()) is None
    assert pass_calls == [[structural_pass]]


def test_agent_config_has_no_zero_stage_offload_or_custom_pass_bans():
    config = CompileConfig(zero3_tuning_strategy="agent",
                           agent_command=["agent"],
                           passes=["z3"],
                           offload_parameters=True)

    assert config.agent_architecture == "two_agent"
    assert config.passes == ["z3"]
    assert config.offload_parameters
    assert not hasattr(config, "validate_zero_stage")


def test_non_agent_default_schedule_retains_fixed_master_passes():
    schedule = init_z3_module._default_z3_schedule(CompileConfig())

    assert schedule == [
        (0, [init_z3_module.zero3_compile.add_z3_gather_release]),
        (init_z3_module.WARMUP, [
            init_z3_module.zero3_compile.add_z3_gather_release, init_z3_module.prefetch.schedule_prefetch,
            init_z3_module.selective_gather.selective_gather
        ]),
    ]


def test_non_agent_explicit_schedule_is_returned_unchanged():
    custom_pass = lambda *args, **kwargs: None
    schedule = [(0, [init_z3_module.zero3_compile.add_z3_gather_release]),
                (init_z3_module.WARMUP, [custom_pass, init_z3_module.prefetch.schedule_prefetch])]
    original = [(step, list(passes)) for step, passes in schedule]

    composed = init_z3_module._compose_agent_schedule(schedule, CompileConfig())

    assert composed is schedule
    assert composed == original


def test_agent_is_appended_after_all_explicit_warmup_passes():
    first_pass = lambda *args, **kwargs: None
    second_pass = lambda *args, **kwargs: None
    schedule = [(0, [init_z3_module.zero3_compile.add_z3_gather_release]),
                (init_z3_module.WARMUP, [first_pass, second_pass])]

    composed = init_z3_module._compose_agent_schedule(schedule, _config())

    assert schedule[-1][1] == [first_pass, second_pass]
    assert composed[-1][1] == [first_pass, second_pass, backend.agent_optimization_loop]


def test_agent_adds_warmup_to_explicit_schedule_without_one():
    structural_pass = lambda *args, **kwargs: None
    later_pass = lambda *args, **kwargs: None
    schedule = [(0, [init_z3_module.zero3_compile.add_z3_gather_release, structural_pass]), (7, [later_pass])]

    composed = init_z3_module._compose_agent_schedule(schedule, _config())

    assert composed == [
        schedule[0],
        (init_z3_module.WARMUP,
         [init_z3_module.zero3_compile.add_z3_gather_release, structural_pass, backend.agent_optimization_loop]),
        schedule[1],
    ]


@pytest.mark.parametrize("offload_kind", ["parameters", "optimizer", "activation"])
def test_agent_schedule_preserves_each_supported_offload_path(offload_kind, monkeypatch):
    options = {
        "offload_parameters": offload_kind == "parameters",
        "offload_opt_states": offload_kind == "optimizer",
        "offload_activation": offload_kind == "activation",
    }
    if offload_kind == "activation":
        monkeypatch.setattr(init_z3_module.offload_activation, "register_activation_offload_ops", lambda: None)
    schedule = init_z3_module._default_z3_schedule(_config(**options))

    assert schedule[0][0] == 0
    assert init_z3_module.zero3_compile.add_z3_gather_release in schedule[0][1]
    assert schedule[-1][0] == init_z3_module.WARMUP
    assert schedule[-1][1][-1] is backend.agent_optimization_loop
    if offload_kind == "parameters":
        assert init_z3_module.offload_parameters.offload_parameter_fwd in schedule[0][1]
        assert init_z3_module.offload_parameters.offload_parameter_fwd in schedule[-1][1]
    elif offload_kind == "optimizer":
        offload_module = importlib.import_module("deepspeed.compile.passes.offload_adam_states")
        assert schedule[1][0] == 1
        assert offload_module.offload_adam_states_for_init in schedule[1][1]
        assert offload_module.offload_adam_states_for_init in schedule[-1][1]
        assert offload_module.move_opt_states in schedule[-1][1]
    else:
        assert init_z3_module.offload_activation.offload_activation_floor in schedule[0][1]
        assert init_z3_module.offload_activation.offload_activation in schedule[-1][1]


def test_agent_timeout_terminates_wrapper_process_group(tmp_path, monkeypatch):
    popen_kwargs = {}
    signals = []

    class FakeProcess:
        pid = 43210
        returncode = None

        def __init__(self):
            self.wait_count = 0

        def wait(self, timeout=None):
            self.wait_count += 1
            if self.wait_count < 3:
                raise subprocess.TimeoutExpired(["agent"], timeout)
            self.returncode = -signal.SIGKILL
            return self.returncode

    process = FakeProcess()

    def fake_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr(agent_runner_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(agent_runner_module.os, "killpg", lambda process_group, sig: signals.append((process_group,
                                                                                                    sig)))
    monkeypatch.setattr(agent_runner_module.time, "sleep", lambda _seconds: None)
    runner = AgentRunner(AgentRunnerConfig(command=["agent"], timeout_sec=1, debug_log=False, terminate_grace_sec=1))

    result = runner.run("prompt", tmp_path)

    assert result.timed_out
    assert popen_kwargs["start_new_session"] is True
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]


def test_agent_timeout_stops_a_real_descendant_process(tmp_path):
    child_pid_path = tmp_path / "child.pid"
    child = (
        "from pathlib import Path; import os, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)")
    wrapper = (
        "from pathlib import Path; import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]]); "
        "deadline = time.monotonic() + 5; "
        "\nwhile not Path(sys.argv[1]).exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
        "time.sleep(60)")
    runner = AgentRunner(
        AgentRunnerConfig(command=[sys.executable, "-c", wrapper, str(child_pid_path), child],
                          timeout_sec=1,
                          debug_log=False,
                          terminate_grace_sec=1))

    result = runner.run("prompt", tmp_path / "real_process_group")

    assert result.timed_out
    assert child_pid_path.exists()
    child_pid = int(child_pid_path.read_text())
    child_stat = Path(f"/proc/{child_pid}/stat")
    for _ in range(50):
        if not child_stat.exists():
            break
        state = child_stat.read_text(encoding="utf-8").split()[2]
        if state in {"X", "Z"}:
            break
        time.sleep(0.02)
    assert not child_stat.exists() or child_stat.read_text(encoding="utf-8").split()[2] in {"X", "Z"}
