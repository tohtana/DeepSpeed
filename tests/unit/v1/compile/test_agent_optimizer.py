# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import copy
from datetime import timedelta
import importlib
import inspect
import json
import multiprocessing
from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace

import pytest
import torch

import deepspeed.compile.backend as backend
import deepspeed.compile.optimizer as optimizer_module
import deepspeed.comm as dist
from deepspeed.compile.config import CompileConfig
from deepspeed.compile.evaluation_context import (GENERATED_PASS_ENTRYPOINT, AgentResponseError, GeneratedPassProposal,
                                                  build_evaluation_packet, build_reference_pass_inventory,
                                                  parse_search_response, serialize_search_context, validate_selection)
from deepspeed.compile.graph_edit import clone_graph_module, structural_fingerprint
from deepspeed.compile.inductor import patch_compiler
from deepspeed.compile.optimizer import (FrozenGraphContext, GraphAgentLoopOptimizer, OptimizationContext,
                                         OptimizationResult, apply_generated_pass, load_generated_pass)
from deepspeed.compile.profilers import ProfilingResult

init_z3_module = importlib.import_module("deepspeed.compile.init_z3")


class ChainModule(torch.nn.Module):

    def forward(self, x):
        return torch.sigmoid(torch.relu(x))


class StaticRunner:

    def __init__(self, callback):
        self.callback = callback
        self.config = SimpleNamespace(command=["agent"])
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
        "agent_architecture": "graph_agent",
        "agent_command": ["agent"],
        "agent_max_iterations": 2,
        "agent_max_retries_per_iteration": 1,
        "agent_timeout_sec": 5,
    }
    values.update(kwargs)
    return CompileConfig(**values)


def _context(config=None):
    config = config or _config()
    gm = torch.fx.symbolic_trace(ChainModule())
    names = [node.name for node in gm.graph.nodes]
    profile = ProfilingResult(fwd_graph=gm.graph,
                              fwd_mem=[(name, 1, 0, 1) for name in names],
                              fwd_time=[(name, 1.0, 1.0) for name in names],
                              fwd_tensor_sizes=[(name, 4) for name in names],
                              fwd_mem_complete=True,
                              needs_backward=False)
    manager = object()
    ctx = OptimizationContext(gm=gm,
                              graph_id=11,
                              graph_slot=(0, "fwd"),
                              graph_order=[(11, False)],
                              profiling_results={11: profile},
                              create_inputs_fn=lambda: (torch.tensor([-2.0, 3.0]), ),
                              mem_budget=1234.0,
                              param_manager=manager,
                              bwd=False,
                              debug_log=False,
                              compile_config=config)
    return gm, ctx


def _fake_profile(gm, ctx, profile_calls=None, expected_abi=None):
    if profile_calls is None:
        profile_calls = optimizer_module._capture_profile_calls(ctx)
    optimizer_module._validate_runtime_call_contract(gm, profile_calls)
    output = gm(*profile_calls[0].args, **profile_calls[0].kwargs)
    runtime_abi = optimizer_module._runtime_abi_descriptor(gm, profile_calls, output)
    if expected_abi is not None and runtime_abi != expected_abi:
        raise optimizer_module._RuntimeABIError("Candidate output ABI differs from the frozen graph")
    profile = copy.deepcopy(ctx.profiling_results[ctx.graph_id])
    names = [node.name for node in gm.graph.nodes]
    for node in gm.graph.nodes:
        node.meta.update({
            "device_time": 2.0,
            "wall_time": 3.0,
            "tensor_size": 4,
            "local_device_time": 1.5,
            "local_wall_time": 2.5,
        })
    profile.fwd_graph = gm.graph
    profile.fwd_mem = [(name, 20, 0, 30) for name in names]
    profile.fwd_time = [(name, 2.0, 3.0) for name in names]
    profile.fwd_tensor_sizes = [(name, 4) for name in names]
    profile.fwd_mem_complete = True
    profile._deepcompile_local_fwd_mem = [(name, 10, 0, 15) for name in names]
    return profile, runtime_abi


NOOP_SOURCE = """def deepcompile_pass(gm, graph_id, graph_order, profiling_results, create_inputs_fn, mem_budget, param_manager, bwd):
    return gm
"""

NEG_SOURCE = """import torch

def deepcompile_pass(gm, graph_id, graph_order, profiling_results, create_inputs_fn, mem_budget, param_manager, bwd):
    assert mem_budget == 1234.0
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.sigmoid:
            node.target = torch.neg
    return gm
"""

COS_SOURCE = """import torch

def deepcompile_pass(gm, graph_id, graph_order, profiling_results, create_inputs_fn, mem_budget, param_manager, bwd):
    if any(node.target is torch.neg for node in gm.graph.nodes if node.op == "call_function"):
        raise RuntimeError("candidate inherited an earlier graph")
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.sigmoid:
            node.target = torch.cos
    return gm
"""


def _evaluate_response(source, summary="candidate"):
    return {
        "schema_version": 1,
        "action": "evaluate",
        "summary": summary,
        "entrypoint": GENERATED_PASS_ENTRYPOINT,
        "source": source,
    }


def _candidate_finish(prompt, index=0):
    record = prompt["history"][index]
    proposal = record["proposal"]
    evaluation = record["evaluation"]
    return {
        "schema_version": 1,
        "action": "finish",
        "summary": f"select {proposal['candidate_id']}",
        "selection": {
            "kind": "candidate",
            "candidate_id": proposal["candidate_id"],
            "source_sha256": proposal["source_sha256"],
            "entrypoint": proposal["entrypoint"],
            "result_fingerprint": evaluation["result_fingerprint"],
        },
    }


def _baseline_finish(prompt):
    return {
        "schema_version": 1,
        "action": "finish",
        "summary": "select baseline graph",
        "selection": {
            "kind": "baseline",
            "frozen_base_fingerprint": prompt["frozen_base"]["fingerprint"],
        },
    }


def test_search_response_requires_complete_source_on_first_turn():
    finish = {
        "schema_version": 1,
        "action": "finish",
        "summary": "premature",
        "selection": {
            "kind": "baseline",
            "frozen_base_fingerprint": "base",
        },
    }
    with pytest.raises(AgentResponseError, match="first coding-agent turn"):
        parse_search_response(json.dumps(finish), 0, [], (0, "fwd"))

    proposal = parse_search_response(json.dumps(_evaluate_response(NOOP_SOURCE)), 0, [], (0, "fwd"))

    assert proposal.candidate_id.startswith("candidate_000_")
    assert len(proposal.source_sha256) == 64
    assert proposal.module_name.endswith(proposal.source_sha256)


def test_identical_source_has_distinct_candidate_and_module_identity():
    first = parse_search_response(json.dumps(_evaluate_response(NOOP_SOURCE)), 0, [], (0, "fwd"))
    second = parse_search_response(json.dumps(_evaluate_response(NOOP_SOURCE)), 1, [], (0, "fwd"))
    other_slot = parse_search_response(json.dumps(_evaluate_response(NOOP_SOURCE)), 0, [], (1, "bwd"))

    assert first.source_sha256 == second.source_sha256 == other_slot.source_sha256
    assert len({first.candidate_id, second.candidate_id}) == 2
    assert len({first.module_name, second.module_name, other_slot.module_name}) == 3


def test_selection_requires_exact_valid_evaluation_identity():
    proposal = parse_search_response(json.dumps(_evaluate_response(NOOP_SOURCE)), 0, [], (0, "fwd"))
    evaluation = {
        "valid": True,
        "result_fingerprint": "result",
        "frozen_base_fingerprint": "base",
    }
    history = [{"proposal": proposal.to_dict(), "evaluation": evaluation}]
    finish = {
        "schema_version": 1,
        "action": "finish",
        "summary": "select candidate",
        "selection": {
            "kind": "candidate",
            "candidate_id": proposal.candidate_id,
            "source_sha256": proposal.source_sha256,
            "entrypoint": proposal.entrypoint,
            "result_fingerprint": "result",
        },
    }

    selection = parse_search_response(json.dumps(finish), 1, history, (0, "fwd"))

    assert validate_selection(selection, history) == proposal
    stale = copy.deepcopy(finish)
    stale["selection"]["result_fingerprint"] = "stale"
    with pytest.raises(AgentResponseError, match="stale result_fingerprint"):
        parse_search_response(json.dumps(stale), 1, history, (0, "fwd"))


def test_reference_inventory_uses_exact_live_source_bytes():
    compile_root = Path(optimizer_module.__file__).resolve().parent
    inventory = build_reference_pass_inventory(compile_root)

    assert len(inventory["files"]) == 12
    assert inventory["source_root"] == str(compile_root)
    assert len(inventory["inventory_sha256"]) == 64
    for record in inventory["files"]:
        path = compile_root / record["path"].removeprefix("deepspeed/compile/")
        assert record["source"] == path.read_text(encoding="utf-8")
        assert len(record["sha256"]) == 64


def test_reference_inventory_includes_move_opt_states_sync():
    inventory = build_reference_pass_inventory()
    offload_adam_states = next(record for record in inventory["files"]
                               if record["path"].endswith("passes/offload_adam_states.py"))
    offload_module = importlib.import_module("deepspeed.compile.passes.offload_adam_states")
    entrypoint = getattr(offload_module, "move_opt_states_sync")

    assert "move_opt_states_sync" in offload_adam_states["entrypoints"]
    assert len(inspect.signature(entrypoint).parameters) == 8


def test_search_prompt_contains_three_complete_reference_sources_and_exact_history():
    gm, ctx = _context()
    frozen = FrozenGraphContext(graph_module=clone_graph_module(gm),
                                graph_fingerprint=structural_fingerprint(gm, ctx.graph_id),
                                graph_slot=ctx.graph_slot,
                                graph_order=ctx.graph_order,
                                baseline_rank_results=[],
                                baseline_aggregate={},
                                mem_budget=ctx.mem_budget,
                                param_manager=ctx.param_manager)
    history = [{"proposal": {"source": "exact prior source"}, "evaluation": {"valid": False}}]

    prompt = json.loads(serialize_search_context(ctx, frozen, build_reference_pass_inventory(), history))

    references = prompt["reference_passes"]["closest_complete_sources"]
    assert len(references) == 3
    assert all(reference["source"] for reference in references)
    assert prompt["history"] == history
    assert prompt["response_contract"]["allowed_actions"] == ["evaluate", "finish"]
    assert "threshold" in " ".join(prompt["response_contract"]["instructions"])


def test_search_context_advertises_nested_finish_selection_shape():
    gm, ctx = _context()
    frozen = FrozenGraphContext(graph_module=clone_graph_module(gm),
                                graph_fingerprint=structural_fingerprint(gm, ctx.graph_id),
                                graph_slot=ctx.graph_slot,
                                graph_order=ctx.graph_order,
                                baseline_rank_results=[],
                                baseline_aggregate={},
                                mem_budget=ctx.mem_budget,
                                param_manager=ctx.param_manager)

    prompt = json.loads(
        serialize_search_context(ctx, frozen, build_reference_pass_inventory(), [], selection_only=True))
    finish = prompt["response_contract"]["finish"]

    assert finish["fields"] == ["schema_version", "action=finish", "summary", "selection"]
    assert finish["baseline_example"] == {
        "schema_version": 1,
        "action": "finish",
        "summary": "why baseline is selected after evaluating candidates",
        "selection": {
            "kind": "baseline",
            "frozen_base_fingerprint": frozen.graph_fingerprint,
        },
    }
    assert finish["candidate_example"]["selection"]["kind"] == "candidate"
    assert any("inside the top-level selection object" in instruction
               for instruction in prompt["response_contract"]["instructions"])


@pytest.mark.parametrize("return_line", ["return None", "return gm"])
def test_generated_pass_loads_exact_bytes_and_accepts_valid_return_contract(tmp_path, return_line):
    gm, ctx = _context()
    source = NOOP_SOURCE.replace("return gm", return_line)
    proposal = parse_search_response(json.dumps(_evaluate_response(source)), 0, [], ctx.graph_slot)
    source_path = tmp_path / "rank_0" / "generated_pass.py"

    candidate, entrypoint = apply_generated_pass(proposal, clone_graph_module(gm), ctx, source_path)

    assert source_path.read_bytes() == source.encode("utf-8")
    assert callable(entrypoint)
    candidate.graph.lint()


@pytest.mark.parametrize("source,phase", [("def deepcompile_pass(:\n", "syntax"),
                                          ("deepcompile_pass = 1\n", "callable"),
                                          ("def deepcompile_pass(gm):\n    return gm\n", "signature"),
                                          (NOOP_SOURCE.replace("return gm", "return object()"), "return_contract")])
def test_generated_pass_reports_mechanical_failure_phase(tmp_path, source, phase):
    gm, ctx = _context()
    proposal = parse_search_response(json.dumps(_evaluate_response(source)), 0, [], ctx.graph_slot)

    with pytest.raises(optimizer_module.GeneratedPassValidationError) as error:
        apply_generated_pass(proposal, clone_graph_module(gm), ctx, tmp_path / "generated.py")

    assert error.value.phase == phase


def test_same_source_candidates_do_not_share_module_globals(tmp_path):
    gm, ctx = _context()
    source = NOOP_SOURCE.replace("def deepcompile_pass", "calls = 0\n\ndef deepcompile_pass").replace(
        "    return gm", "    global calls\n    calls += 1\n    return gm")
    first = parse_search_response(json.dumps(_evaluate_response(source)), 0, [], ctx.graph_slot)
    second = parse_search_response(json.dumps(_evaluate_response(source)), 1, [], ctx.graph_slot)
    first_fn = load_generated_pass(first, tmp_path / "first.py")
    second_fn = load_generated_pass(second, tmp_path / "second.py")

    apply_generated_pass(first, clone_graph_module(gm), ctx, tmp_path / "first.py", first_fn)

    assert first_fn.__globals__["calls"] == 1
    assert second_fn.__globals__["calls"] == 0


def test_generated_pass_can_install_graph_referenced_modules_without_allowlist(tmp_path):
    gm, ctx = _context()
    source = """import torch

def deepcompile_pass(gm, graph_id, graph_order, profiling_results, create_inputs_fn, mem_budget, param_manager, bwd):
    gm.add_module("generated_relu", torch.nn.ReLU())
    output = next(node for node in gm.graph.nodes if node.op == "output")
    with gm.graph.inserting_before(output):
        result = gm.graph.call_module("generated_relu", (output.args[0],))
    output.args = (result,)
    return gm
"""
    proposal = parse_search_response(json.dumps(_evaluate_response(source)), 0, [], ctx.graph_slot)

    candidate, _ = apply_generated_pass(proposal, clone_graph_module(gm), ctx, tmp_path / "generated_module.py")

    assert isinstance(candidate.generated_relu, torch.nn.ReLU)
    assert any(node.op == "call_module" and node.target == "generated_relu" for node in candidate.graph.nodes)


def test_evaluation_packet_is_observational_and_sanitizes_nonfinite_metrics():
    proposal = parse_search_response(json.dumps(_evaluate_response(NOOP_SOURCE)), 0, [], (0, "fwd"))
    rank_results = [{
        "rank": 0,
        "success": True,
        "local_device_time": float("nan"),
        "local_peak_memory": 4,
        "error": None,
    }]

    packet = build_evaluation_packet(proposal, rank_results, {}, "base", {"runtime_abi": {"success": True}})

    assert packet["valid"]
    assert "accept" not in packet and "reject" not in packet and "winner" not in packet
    assert packet["rank_results"][0]["local_device_time"] == {"non_finite_float": "nan"}
    assert packet["aggregate"]["device_time"]["mean"] is None
    assert packet["correctness_available"] is False


def test_two_complete_candidates_start_fresh_and_agent_selects_earlier_source(monkeypatch, tmp_path):
    gm, ctx = _context(_config(agent_max_iterations=2))

    def callback(prompt, role, call_index):
        assert role == "coding_agent"
        if call_index == 0:
            return _evaluate_response(NEG_SOURCE, "candidate one")
        if call_index == 1:
            assert prompt["history"][0]["proposal"]["source"] == NEG_SOURCE
            assert prompt["history"][0]["evaluation"]["valid"]
            return _evaluate_response(COS_SOURCE, "candidate two from frozen base")
        assert prompt["selection_only"]
        assert len(prompt["history"]) == 2
        return _candidate_finish(prompt, 0)

    runner = StaticRunner(callback)
    monkeypatch.setenv("DEEPCOMPILE_AGENT_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(optimizer_module, "_profile_graph", _fake_profile)
    monkeypatch.setattr(optimizer_module, "_cleanup_after_candidate", lambda: None)
    optimizer_module._reset_inspection_session_root()

    result = GraphAgentLoopOptimizer(runner, ctx.compile_config).optimize(gm, ctx)

    assert [entry.action for entry in result.trace if entry.action == "evaluated"] == ["evaluated", "evaluated"]
    assert result.trace[-1].action == "selected"
    assert any(node.target is torch.neg for node in gm.graph.nodes)
    assert not any(node.target is torch.cos for node in gm.graph.nodes)
    assert len(runner.calls) == 3
    session = next(tmp_path.iterdir())
    assert (session / "graph_0_fwd" / "candidate_000" / "generated_pass.py").read_text() == NEG_SOURCE


def test_invalid_candidate_packet_can_lead_to_another_proposal(monkeypatch):
    gm, ctx = _context(_config(agent_max_iterations=2))
    invalid_source = "def deepcompile_pass(:\n"

    def callback(prompt, _role, call_index):
        if call_index == 0:
            return _evaluate_response(invalid_source, "invalid syntax")
        if call_index == 1:
            assert prompt["history"][0]["proposal"]["source"] == invalid_source
            assert not prompt["history"][0]["evaluation"]["valid"]
            return _evaluate_response(NOOP_SOURCE, "try another source")
        return _baseline_finish(prompt)

    runner = StaticRunner(callback)
    monkeypatch.setattr(optimizer_module, "_profile_graph", _fake_profile)
    monkeypatch.setattr(optimizer_module, "_cleanup_after_candidate", lambda: None)

    result = GraphAgentLoopOptimizer(runner, ctx.compile_config).optimize(gm, ctx)

    evaluated = [entry for entry in result.trace if entry.action == "evaluated"]
    assert [entry.details["valid"] for entry in evaluated] == [False, True]
    assert result.trace[-1].action == "selected"
    assert len(runner.calls) == 3


def test_downstream_backend_compile_failure_rejects_candidate_before_profiling(tmp_path):
    gm, ctx = _context()
    frozen_graph = clone_graph_module(gm)
    frozen = FrozenGraphContext(graph_module=frozen_graph,
                                graph_fingerprint=structural_fingerprint(frozen_graph, ctx.graph_id),
                                graph_slot=ctx.graph_slot,
                                graph_order=ctx.graph_order,
                                baseline_rank_results=[],
                                baseline_aggregate={},
                                mem_budget=ctx.mem_budget,
                                param_manager=ctx.param_manager)
    compiled_graphs = []

    def fail_backend_compile(candidate):
        compiled_graphs.append(candidate)
        raise AssertionError("both a fallback and a decomp for same op: aten.silu.default")

    ctx.backend_compile_fn = fail_backend_compile
    proposal = parse_search_response(json.dumps(_evaluate_response(NOOP_SOURCE)), 0, [], ctx.graph_slot)

    _, candidate_profile, _, packet = GraphAgentLoopOptimizer._apply_and_profile_candidate(
        ctx, frozen, proposal, 0, tmp_path / "generated.py")

    backend_validation = packet["validation"]["backend_compile"]
    assert not packet["valid"]
    assert candidate_profile is None
    assert len(compiled_graphs) == 1
    assert compiled_graphs[0] is not gm and compiled_graphs[0] is not frozen_graph
    assert not backend_validation["success"]
    assert not backend_validation["rank_results"][0]["skipped"]
    assert "both a fallback and a decomp" in backend_validation["rank_results"][0]["error"]


def test_agent_can_finish_after_exactly_one_evaluated_candidate(monkeypatch):
    gm, ctx = _context(_config(agent_max_iterations=3))

    def callback(prompt, _role, call_index):
        if call_index == 0:
            return _evaluate_response(NOOP_SOURCE)
        assert not prompt["selection_only"]
        return _baseline_finish(prompt)

    runner = StaticRunner(callback)
    monkeypatch.setattr(optimizer_module, "_profile_graph", _fake_profile)
    monkeypatch.setattr(optimizer_module, "_cleanup_after_candidate", lambda: None)

    result = GraphAgentLoopOptimizer(runner, ctx.compile_config).optimize(gm, ctx)

    assert len(runner.calls) == 2
    assert result.trace[-1].action == "selected"
    assert result.trace[-1].details["live_source_executed"] is False


def test_invalid_selection_response_aborts_without_implicit_baseline(monkeypatch):
    gm, ctx = _context(_config(agent_max_iterations=1, agent_max_retries_per_iteration=0))
    original = structural_fingerprint(gm, ctx.graph_id)

    def callback(prompt, _role, call_index):
        if call_index == 0:
            return _evaluate_response(NOOP_SOURCE)
        response = _candidate_finish(prompt)
        response["selection"]["result_fingerprint"] = "stale"
        return response

    runner = StaticRunner(callback)
    monkeypatch.setattr(optimizer_module, "_profile_graph", _fake_profile)
    monkeypatch.setattr(optimizer_module, "_cleanup_after_candidate", lambda: None)

    result = GraphAgentLoopOptimizer(runner, ctx.compile_config).optimize(gm, ctx)

    assert result.trace[-1].action == "abort"
    assert "no selection made" in result.trace[-1].summary
    assert structural_fingerprint(gm, ctx.graph_id) == original
    assert not any(entry.action == "selected" for entry in result.trace)


def test_final_live_failure_restores_graph_and_raises_fail_closed(monkeypatch):
    gm, ctx = _context(_config(agent_max_iterations=1))
    source = """import torch

calls = 0

def deepcompile_pass(gm, graph_id, graph_order, profiling_results, create_inputs_fn, mem_budget, param_manager, bwd):
    global calls
    calls += 1
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.sigmoid:
            node.target = torch.neg
    if calls == 2:
        raise RuntimeError("fail during live replay")
    return gm
"""

    def callback(prompt, _role, call_index):
        return _evaluate_response(source) if call_index == 0 else _candidate_finish(prompt)

    runner = StaticRunner(callback)
    monkeypatch.setattr(optimizer_module, "_profile_graph", _fake_profile)
    monkeypatch.setattr(optimizer_module, "_cleanup_after_candidate", lambda: None)

    with pytest.raises(RuntimeError, match="after live source execution"):
        GraphAgentLoopOptimizer(runner, ctx.compile_config).optimize(gm, ctx)

    assert any(node.target is torch.sigmoid for node in gm.graph.nodes)
    assert not any(node.target is torch.neg for node in gm.graph.nodes)


def test_nonzero_rank_never_invokes_coding_agent(monkeypatch):
    gm, ctx = _context()
    runner = StaticRunner(lambda *_args: pytest.fail("nonzero rank invoked coding agent"))
    monkeypatch.setattr(optimizer_module, "_rank", lambda: 1)
    monkeypatch.setattr(GraphAgentLoopOptimizer, "_snapshot_consensus", lambda _self, _ctx: (False, []))

    result = GraphAgentLoopOptimizer(runner, ctx.compile_config).optimize(gm, ctx)

    assert result.trace == []
    assert runner.calls == []


def test_structural_consensus_failure_gathers_normalized_rank_graphs(monkeypatch):
    gm, ctx = _context()
    first_records = [{
        "rank": 0,
        "success": True,
        "fingerprint": "rank-zero",
        "error": None,
    }, {
        "rank": 1,
        "success": True,
        "fingerprint": "rank-one",
        "error": None,
    }]
    gathered = []

    def fake_gather(local):
        gathered.append(local)
        if len(gathered) == 1:
            return first_records
        return [{**local, "rank": rank} for rank in range(2)]

    monkeypatch.setattr(optimizer_module, "gather_rank_records", fake_gather)

    consensus, records = GraphAgentLoopOptimizer._snapshot_consensus(ctx)

    assert not consensus
    assert len(gathered) == 2
    assert records[0]["normalized_graph"] == optimizer_module.normalized_graph(gm, runtime_graph_id=ctx.graph_id)
    assert records[0]["description_error"] is None


def test_pre_inventory_abort_is_persisted_with_guard_and_rank_evidence(monkeypatch, tmp_path):
    gm, ctx = _context()
    records = [{
        "rank": 0,
        "success": True,
        "fingerprint": "rank-zero",
        "error": None,
        "normalized_graph": [{
            "id": "base:0"
        }],
        "description_error": None,
    }, {
        "rank": 1,
        "success": True,
        "fingerprint": "rank-one",
        "error": None,
        "normalized_graph": [{
            "id": "base:1"
        }],
        "description_error": None,
    }]
    runner = StaticRunner(lambda *_args: pytest.fail("early abort invoked coding agent"))
    monkeypatch.setenv("DEEPCOMPILE_AGENT_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(GraphAgentLoopOptimizer, "_snapshot_consensus", lambda _self, _ctx: (False, records))
    optimizer_module._reset_inspection_session_root()

    result = GraphAgentLoopOptimizer(runner, ctx.compile_config).optimize(gm, ctx)

    evidence_path = next(tmp_path.iterdir()) / "graph_0_fwd" / "early_abort.json"
    evidence = json.loads(evidence_path.read_text())
    assert result.trace[-1].action == "abort"
    assert evidence["guard"] == "post_z3_structural_consensus"
    assert evidence["graph_slot"] == [0, "fwd"]
    assert evidence["rank_results"] == records
    assert runner.calls == []


def test_backend_passes_real_mem_budget_and_param_manager_to_private_context(monkeypatch):
    gm, ctx = _context()
    manager = object()
    backend_compile_fn = object()
    observed = []

    class FakeOptimizer:

        def __init__(self, _runner, _config):
            pass

        def optimize(self, graph_module, optimization_context):
            observed.append((graph_module, optimization_context.mem_budget, optimization_context.param_manager,
                             optimization_context.backend_compile_fn))
            return OptimizationResult()

    monkeypatch.setattr(backend, "run_opt_passes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(optimizer_module, "GraphAgentLoopOptimizer", FakeOptimizer)

    result = backend.run_optimization([backend.agent_optimization_loop],
                                      gm,
                                      11, (0, "fwd"),
                                      ctx.graph_order,
                                      ctx.profiling_results,
                                      ctx.create_inputs_fn,
                                      987.0,
                                      manager,
                                      False,
                                      ctx.compile_config,
                                      backend_compile_fn=backend_compile_fn)

    assert isinstance(result, OptimizationResult)
    assert observed == [(gm, 987.0, manager, backend_compile_fn)]


def test_inductor_wrapper_exposes_exact_downstream_compiler_and_inputs():
    gm, _ = _context()
    fake_inputs = (object(), )
    compile_calls = []
    tracing_context = torch._guards.TracingContext(None)
    tracing_context.output_strides = []

    def original_compiler(candidate, compiler_inputs):
        compile_calls.append((candidate, compiler_inputs))
        torch._guards.TracingContext.get().output_strides.append(candidate is gm)
        return lambda: candidate

    def deepcompile_compiler(graph_module, sample_inputs, backend_compile_fn):
        assert graph_module is gm
        assert sample_inputs is fake_inputs
        candidate = clone_graph_module(graph_module)
        assert callable(backend_compile_fn(candidate))
        return graph_module.graph

    wrapped = patch_compiler(original_compiler,
                             deepcompile_compiler,
                             z3_partition=False,
                             graph_id=11,
                             graph_param_manager={},
                             bwd=False)

    with torch._guards.tracing(tracing_context):
        result = wrapped(gm, fake_inputs)

    assert callable(result)
    assert [record[1] for record in compile_calls] == [fake_inputs, fake_inputs]
    assert compile_calls[0][0] is not gm
    assert compile_calls[1][0] is gm
    assert tracing_context.output_strides == [True]


def test_required_z3_and_offload_passes_still_precede_agent_marker(monkeypatch):
    monkeypatch.setattr(init_z3_module.offload_activation, "register_activation_offload_ops", lambda: None)
    for options in ({}, {"offload_parameters": True}, {"offload_opt_states": True}, {"offload_activation": True}):
        schedule = init_z3_module._default_z3_schedule(_config(**options))
        warmup_passes = schedule[-1][1]
        assert warmup_passes[-1] is backend.agent_optimization_loop
        assert init_z3_module.zero3_compile.add_z3_gather_release in warmup_passes


def _rank_divergent_source_worker(rank, init_method, result_queue):
    try:
        dist.init_distributed(dist_backend="gloo",
                              auto_mpi_discovery=False,
                              init_method=init_method,
                              rank=rank,
                              world_size=2,
                              timeout=timedelta(seconds=10),
                              verbose=False)
        gm, ctx = _context(_config())
        frozen_graph = clone_graph_module(gm)
        fingerprint = structural_fingerprint(frozen_graph, ctx.graph_id)
        frozen = FrozenGraphContext(graph_module=frozen_graph,
                                    graph_fingerprint=fingerprint,
                                    graph_slot=ctx.graph_slot,
                                    graph_order=ctx.graph_order,
                                    baseline_rank_results=[],
                                    baseline_aggregate={},
                                    mem_budget=ctx.mem_budget,
                                    param_manager=ctx.param_manager)
        source = """import deepspeed.comm as dist
import torch

def deepcompile_pass(gm, graph_id, graph_order, profiling_results, create_inputs_fn, mem_budget, param_manager, bwd):
    if dist.get_rank() == 0:
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is torch.sigmoid:
                node.target = torch.neg
    return gm
"""
        local_proposal = None
        if rank == 0:
            local_proposal = parse_search_response(json.dumps(_evaluate_response(source)), 0, [], ctx.graph_slot)
        envelope = optimizer_module.broadcast_json_payload({"proposal": local_proposal.to_dict()} if rank ==
                                                           0 else None)
        proposal = GeneratedPassProposal(**envelope["proposal"])
        source_path = Path(tempfile.mkdtemp(prefix=f"deepcompile_rank_divergent_{rank}_")) / "generated_pass.py"
        _, _, _, packet = GraphAgentLoopOptimizer._apply_and_profile_candidate(ctx, frozen, proposal, 0, source_path)
        result_queue.put({"rank": rank, "valid": packet["valid"], "error": None})
    except Exception as exc:
        result_queue.put({"rank": rank, "valid": None, "error": repr(exc)})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_two_rank_source_consensus_rejects_rank_dependent_graph(tmp_path):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    init_method = f"file://{tmp_path / 'gloo_init'}"
    processes = [
        context.Process(target=_rank_divergent_source_worker, args=(rank, init_method, result_queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = []
    for _ in processes:
        try:
            results.append(result_queue.get(timeout=20))
        except queue.Empty:
            break
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert len(results) == 2
    assert all(result["error"] is None for result in results), results
    assert all(result["valid"] is False for result in results)
