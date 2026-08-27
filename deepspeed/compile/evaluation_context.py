# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from .graph_edit import RUNTIME_GRAPH_ID, normalized_graph, normalized_graph_order

GENERATED_PASS_SCHEMA_VERSION = 1
GENERATED_PASS_ENTRYPOINT = "deepcompile_pass"
BASE_COMMIT_INFORMATIONAL = "55a0d29ac5292fab89c03eece4dab77195b37a17"

_REFERENCE_PASS_FILES = (
    "passes/contract.py",
    "passes/zero3_compile.py",
    "passes/prefetch.py",
    "passes/selective_gather.py",
    "passes/offload_activation.py",
    "passes/offload_parameters.py",
    "passes/offload_adam_states.py",
    "passes/long_context_checkpointing.py",
    "passes/zero_1_and_2_compile.py",
    "passes/tp_compile.py",
    "passes/sp_compile.py",
    "list_schedule.py",
)
_REFERENCE_ENTRYPOINTS = {
    "passes/contract.py": ["validate_schedule"],
    "passes/zero3_compile.py": ["add_z3_gather_release"],
    "passes/prefetch.py": ["schedule_prefetch"],
    "passes/selective_gather.py": ["selective_gather"],
    "passes/offload_activation.py": ["offload_activation", "offload_activation_floor"],
    "passes/offload_parameters.py": ["offload_parameter_fwd"],
    "passes/offload_adam_states.py": ["offload_adam_states_for_init", "move_opt_states", "move_opt_states_sync"],
    "passes/long_context_checkpointing.py": ["register_long_context_checkpointing"],
    "passes/zero_1_and_2_compile.py": ["add_z1_reduce"],
    "passes/tp_compile.py": ["apply_autotp"],
    "passes/sp_compile.py": ["apply_autosp"],
    "list_schedule.py": ["list_schedule", "simple_prefetch", "fast_free_schedule"],
}
_PROMPT_REFERENCE_FILES = (
    "passes/zero3_compile.py",
    "passes/prefetch.py",
    "passes/selective_gather.py",
)


class AgentResponseError(ValueError):
    pass


def extract_json_object(raw_stdout: str) -> str:
    text = raw_stdout.strip()
    if not text:
        raise AgentResponseError("Agent returned empty output")
    if "```" in text:
        for block in text.split("```"):
            candidate = block.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate

    start = text.find("{")
    if start < 0:
        raise AgentResponseError("Agent output does not contain a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise AgentResponseError("Agent output contains an unterminated JSON object")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload_hash(payload: Any) -> str:
    return _sha256_text(_canonical_json(payload))


@dataclass(frozen=True)
class GeneratedPassProposal:
    candidate_id: str
    summary: str
    entrypoint: str
    source: str
    source_sha256: str
    module_name: str
    proposal_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneratedPassSelection:
    summary: str
    kind: str
    candidate_id: Optional[str] = None
    source_sha256: Optional[str] = None
    entrypoint: Optional[str] = None
    frozen_base_fingerprint: Optional[str] = None
    result_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build_proposal(summary: str, source: str, graph_slot: Tuple[int, str], proposal_index: int):
    source_sha256 = _sha256_text(source)
    candidate_id = f"candidate_{proposal_index:03d}_{source_sha256[:8]}"
    slot = re.sub(r"[^0-9A-Za-z_]", "_", f"{graph_slot[0]}_{graph_slot[1]}")
    module_name = f"deepcompile_generated_pass_{slot}_{proposal_index:03d}_{source_sha256}"
    identity = {
        "candidate_id": candidate_id,
        "entrypoint": GENERATED_PASS_ENTRYPOINT,
        "module_name": module_name,
        "source": source,
    }
    return GeneratedPassProposal(candidate_id=candidate_id,
                                 summary=summary,
                                 entrypoint=GENERATED_PASS_ENTRYPOINT,
                                 source=source,
                                 source_sha256=source_sha256,
                                 module_name=module_name,
                                 proposal_hash=_payload_hash(identity))


def verify_proposal_identity(proposal: GeneratedPassProposal, graph_slot: Tuple[int, str],
                             proposal_index: int) -> None:
    expected = _build_proposal(proposal.summary, proposal.source, graph_slot, proposal_index)
    if proposal != expected:
        raise ValueError(f"Generated-pass identity mismatch for {proposal.candidate_id}")


def _required_string(payload: Dict[str, Any], name: str, allow_empty: bool = False) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise AgentResponseError(f"{name} must be a {'string' if allow_empty else 'non-empty string'}")
    return value


def _parse_selection(payload: Any, summary: str) -> GeneratedPassSelection:
    if not isinstance(payload, dict):
        raise AgentResponseError("finish.selection must be an object")
    kind = payload.get("kind")
    if kind == "baseline":
        if set(payload) != {"kind", "frozen_base_fingerprint"}:
            raise AgentResponseError("baseline selection requires only kind and frozen_base_fingerprint")
        return GeneratedPassSelection(summary=summary,
                                      kind=kind,
                                      frozen_base_fingerprint=_required_string(payload, "frozen_base_fingerprint"))
    if kind == "candidate":
        expected_fields = {"kind", "candidate_id", "source_sha256", "entrypoint", "result_fingerprint"}
        if set(payload) != expected_fields:
            raise AgentResponseError(f"candidate selection requires exactly {sorted(expected_fields)}")
        entrypoint = _required_string(payload, "entrypoint")
        if entrypoint != GENERATED_PASS_ENTRYPOINT:
            raise AgentResponseError(f"entrypoint must be '{GENERATED_PASS_ENTRYPOINT}'")
        return GeneratedPassSelection(summary=summary,
                                      kind=kind,
                                      candidate_id=_required_string(payload, "candidate_id"),
                                      source_sha256=_required_string(payload, "source_sha256"),
                                      entrypoint=entrypoint,
                                      result_fingerprint=_required_string(payload, "result_fingerprint"))
    raise AgentResponseError("selection.kind must be 'baseline' or 'candidate'")


def parse_search_response(raw_stdout: str, evaluated_count: int, history: List[Dict[str, Any]],
                          graph_slot: Tuple[int, str]) -> Union[GeneratedPassProposal, GeneratedPassSelection]:
    try:
        payload = json.loads(extract_json_object(raw_stdout))
    except json.JSONDecodeError as exc:
        raise AgentResponseError(f"Coding agent returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AgentResponseError("Coding-agent response must be an object")
    if payload.get("schema_version") != GENERATED_PASS_SCHEMA_VERSION:
        raise AgentResponseError(f"Unsupported generated-pass schema_version {payload.get('schema_version')}")
    summary = _required_string(payload, "summary", allow_empty=True)
    action = payload.get("action")
    if action == "evaluate":
        if set(payload) != {"schema_version", "action", "summary", "entrypoint", "source"}:
            raise AgentResponseError("evaluate requires exactly schema_version, action, summary, entrypoint, source")
        if payload.get("entrypoint") != GENERATED_PASS_ENTRYPOINT:
            raise AgentResponseError(f"entrypoint must be '{GENERATED_PASS_ENTRYPOINT}'")
        source = _required_string(payload, "source")
        return _build_proposal(summary, source, graph_slot, evaluated_count)
    if action == "finish":
        if evaluated_count == 0:
            raise AgentResponseError("The first coding-agent turn must evaluate one complete pass source")
        if set(payload) != {"schema_version", "action", "summary", "selection"}:
            raise AgentResponseError("finish requires exactly schema_version, action, summary, and selection")
        selection = _parse_selection(payload.get("selection"), summary)
        validate_selection(selection, history)
        return selection
    raise AgentResponseError("action must be 'evaluate' or 'finish'")


def validate_selection(selection: GeneratedPassSelection,
                       history: List[Dict[str, Any]],
                       frozen_base_fingerprint: Optional[str] = None) -> Optional[GeneratedPassProposal]:
    if selection.kind == "baseline":
        expected = frozen_base_fingerprint
        if expected is None and history:
            expected = history[0]["evaluation"].get("frozen_base_fingerprint")
        if expected is None or selection.frozen_base_fingerprint != expected:
            raise AgentResponseError("Baseline selection does not match the exact frozen-base fingerprint")
        return None

    for record in history:
        proposal_payload = record.get("proposal", {})
        evaluation = record.get("evaluation", {})
        if proposal_payload.get("candidate_id") != selection.candidate_id:
            continue
        required = {
            "source_sha256": selection.source_sha256,
            "entrypoint": selection.entrypoint,
        }
        for field, expected in required.items():
            if proposal_payload.get(field) != expected:
                raise AgentResponseError(f"Selected candidate has a stale {field}")
        if not evaluation.get("valid"):
            raise AgentResponseError("Selected candidate was not mechanically valid on every rank")
        if evaluation.get("result_fingerprint") != selection.result_fingerprint:
            raise AgentResponseError("Selected candidate has a stale result_fingerprint")
        return GeneratedPassProposal(**proposal_payload)
    raise AgentResponseError(f"Selected candidate '{selection.candidate_id}' was not evaluated in this graph slot")


def build_reference_pass_inventory(source_root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent if source_root is None else Path(source_root).resolve()
    files = []
    for relative_path in _REFERENCE_PASS_FILES:
        path = root / relative_path
        source = path.read_text(encoding="utf-8")
        files.append({
            "path": f"deepspeed/compile/{relative_path}",
            "sha256": _sha256_text(source),
            "entrypoints": list(_REFERENCE_ENTRYPOINTS[relative_path]),
            "source": source,
        })
    inventory = {
        "schema_version": GENERATED_PASS_SCHEMA_VERSION,
        "source_root": str(root),
        "base_commit_informational": BASE_COMMIT_INFORMATIONAL,
        "files": files,
    }
    inventory["inventory_sha256"] = _payload_hash(inventory)
    return inventory


def _profile_payload(profile, bwd: bool) -> Dict[str, Any]:
    times = profile.bwd_time if bwd else profile.fwd_time
    memory = profile.bwd_mem if bwd else profile.fwd_mem
    tensor_sizes = dict(profile.bwd_tensor_sizes if bwd else profile.fwd_tensor_sizes)
    return {
        "total_device_time":
        sum(row[1] for row in times),
        "peak_memory":
        max([row[3] for row in memory], default=0),
        "nodes": [{
            "name_hint": name,
            "device_time": device_time,
            "wall_time": wall_time,
            "tensor_size": tensor_sizes.get(name, 0),
        } for name, device_time, wall_time in times],
    }


def _inventory_prompt_context(inventory: Dict[str, Any]) -> Dict[str, Any]:
    manifest = [{key: value for key, value in item.items() if key != "source"} for item in inventory["files"]]
    prompt_paths = {f"deepspeed/compile/{path}" for path in _PROMPT_REFERENCE_FILES}
    examples = [{
        "path": item["path"],
        "sha256": item["sha256"],
        "source": item["source"],
    } for item in inventory["files"] if item["path"] in prompt_paths]
    return {
        "source_root": inventory["source_root"],
        "inventory_sha256": inventory["inventory_sha256"],
        "manifest": manifest,
        "closest_complete_sources": examples,
    }


def serialize_search_context(ctx,
                             frozen,
                             inventory: Dict[str, Any],
                             history: List[Dict[str, Any]],
                             mechanical_feedback: Optional[List[str]] = None,
                             selection_only: bool = False) -> str:
    evaluated_count = len(history)
    if selection_only:
        allowed_actions = ["finish"]
    elif evaluated_count == 0:
        allowed_actions = ["evaluate"]
    else:
        allowed_actions = ["evaluate", "finish"]
    payload = {
        "role": "deepcompile_coding_agent",
        "objective": "Write complete Python source for an experimental DeepCompile optimization pass, then use "
        "observed evaluations to revise the complete source or explicitly select a result.",
        "graph_slot": {
            "index": ctx.graph_slot[0],
            "direction": ctx.graph_slot[1],
        },
        "frozen_base": {
            "fingerprint": frozen.graph_fingerprint,
            "nodes": normalized_graph(frozen.graph_module, include_hints=True, runtime_graph_id=ctx.graph_id),
            "profile": _profile_payload(ctx.profiling_results[ctx.graph_id], ctx.bwd),
            "baseline_rank_results": frozen.baseline_rank_results,
        },
        "graph_runtime": {
            "graph_id": dict(RUNTIME_GRAPH_ID),
            "graph_order": normalized_graph_order(ctx.graph_order, ctx.graph_id),
            "direction": "bwd" if ctx.bwd else "fwd",
            "mem_budget": ctx.mem_budget,
            "param_manager_type": type(ctx.param_manager).__module__ + "." + type(ctx.param_manager).__qualname__,
        },
        "required_pass_trace": ctx.warmup_trace,
        "reference_passes": _inventory_prompt_context(inventory),
        "history": history,
        "mechanical_feedback": list(mechanical_feedback or []),
        "selection_only": selection_only,
        "response_contract": {
            "schema_version":
            GENERATED_PASS_SCHEMA_VERSION,
            "allowed_actions":
            allowed_actions,
            "evaluate": {
                "fields": ["schema_version", "action=evaluate", "summary", "entrypoint", "source"],
                "entrypoint": GENERATED_PASS_ENTRYPOINT,
                "source": "complete Python file source, not a patch or edit sequence",
            },
            "finish": {
                "baseline": ["kind=baseline", "frozen_base_fingerprint"],
                "candidate": ["kind=candidate", "candidate_id", "source_sha256", "entrypoint", "result_fingerprint"],
            },
            "instructions": [
                "Return exactly one JSON object and no prose.",
                "The first turn must evaluate one complete source. Later turns may evaluate another complete source "
                "or finish; there is no minimum of two candidates.",
                "Each source runs from a fresh clone of this frozen base. Revised source must contain every change "
                "you want evaluated; candidates do not accumulate.",
                "The fixed deepcompile_pass callable receives gm, graph_id, graph_order, profiling_results, "
                "create_inputs_fn, mem_budget, param_manager, and bwd. Return None or the identical gm.",
                "Do not call collectives in the pass body. Rank-dependent graph rewrites fail fingerprint consensus; "
                "inserting collective nodes into the graph is allowed.",
                "Evaluation reports observations and has no automatic latency threshold or correctness oracle.",
                "Generated source is trusted code with no sandbox or import/operator allowlist.",
                "Graph cloning does not roll back arbitrary Python, module-global, filesystem, native, param_manager, "
                "or rank-divergent collective side effects. The selected pass body executes again on the live graph.",
            ],
        },
    }
    return json.dumps(sanitize_json_value(payload), indent=2, default=str, allow_nan=False)


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite_float": str(value)}
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]
    return value


def aggregate_rank_metrics(rank_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [record for record in rank_results if record.get("success")]
    device_times = [
        record["local_device_time"] for record in successful
        if isinstance(record.get("local_device_time"), (int, float))
    ]
    peak_memory = [
        record["local_peak_memory"] for record in successful
        if isinstance(record.get("local_peak_memory"), (int, float))
    ]
    return {
        "device_time": {
            "mean": sum(device_times) / len(device_times) if device_times else None,
            "min": min(device_times) if device_times else None,
            "max": max(device_times) if device_times else None,
        },
        "peak_memory": {
            "max": max(peak_memory) if peak_memory else None,
        },
    }


def build_evaluation_packet(proposal: GeneratedPassProposal, rank_results: List[Dict[str, Any]],
                            baseline_aggregate: Dict[str, Any], frozen_base_fingerprint: str,
                            validation: Dict[str, Any]) -> Dict[str, Any]:
    rank_results = sanitize_json_value(rank_results)
    valid = bool(rank_results) and all(record.get("success") for record in rank_results)
    packet = {
        "schema_version":
        GENERATED_PASS_SCHEMA_VERSION,
        "candidate_id":
        proposal.candidate_id,
        "source_sha256":
        proposal.source_sha256,
        "entrypoint":
        proposal.entrypoint,
        "module_name":
        proposal.module_name,
        "proposal_hash":
        proposal.proposal_hash,
        "result_fingerprint":
        None,
        "frozen_base_fingerprint":
        frozen_base_fingerprint,
        "valid":
        valid,
        "validation":
        sanitize_json_value(validation),
        "rank_results":
        rank_results,
        "aggregate":
        aggregate_rank_metrics(rank_results),
        "baseline_aggregate":
        sanitize_json_value(baseline_aggregate),
        "measurement_method": {
            "timing": "one profiling run per rank with freshly captured inputs; local readings captured before "
            "the existing per-node AVG reductions",
            "memory": "one profiling run per rank with freshly captured inputs; local readings captured before "
            "the existing per-node MAX reductions",
            "rank_spread": "device_time min/max describe ranks, not repeated-run variance",
        },
        "correctness_available":
        False,
        "side_effect_boundary":
        "Only graph topology and covered tensor state are restored between candidates; "
        "arbitrary Python, files, globals, native state, param_manager, and pass-body collectives are not generally "
        "rollback-safe.",
    }
    fingerprint_payload = dict(packet)
    fingerprint_payload.pop("result_fingerprint")
    packet["result_fingerprint"] = _payload_hash(fingerprint_payload)
    return packet
