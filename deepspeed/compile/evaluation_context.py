# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, List, Optional

from torch.fx import GraphModule

from .graph_edit import RUNTIME_GRAPH_ID, normalized_graph, normalized_graph_order, structural_fingerprint


EVALUATION_SCHEMA_VERSION = 3


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


@dataclass(frozen=True)
class GraphSlotRef:
    index: int
    direction: str


@dataclass(frozen=True)
class GraphSnapshotRef:
    graph_slot: GraphSlotRef
    generation: int
    graph_fingerprint: str


class GraphVersionTracker:

    def __init__(self, slot: GraphSlotRef, gm: GraphModule, runtime_graph_id: int):
        self._slot = slot
        self._generation = 0
        self._gm = gm
        self._runtime_graph_id = runtime_graph_id

    def current_ref(self) -> GraphSnapshotRef:
        return GraphSnapshotRef(graph_slot=self._slot,
                                generation=self._generation,
                                graph_fingerprint=structural_fingerprint(self._gm, self._runtime_graph_id))

    def accept(self, gm: GraphModule) -> GraphSnapshotRef:
        self._generation += 1
        self._gm = gm
        return self.current_ref()


@dataclass
class EvaluationDecision:
    schema_version: int
    based_on: GraphSnapshotRef
    decision: str
    summary: str
    optimizer_brief: str = ""
    candidate_generation: Optional[int] = None
    candidate_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _slot_from_dict(payload: Any) -> GraphSlotRef:
    if not isinstance(payload, dict):
        raise AgentResponseError("graph_slot must be an object")
    index = payload.get("index")
    direction = payload.get("direction")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise AgentResponseError("graph_slot.index must be a non-negative integer")
    if direction not in {"fwd", "bwd"}:
        raise AgentResponseError("graph_slot.direction must be 'fwd' or 'bwd'")
    return GraphSlotRef(index=index, direction=direction)


def graph_snapshot_from_dict(payload: Any) -> GraphSnapshotRef:
    if not isinstance(payload, dict):
        raise AgentResponseError("based_on must be an object")
    generation = payload.get("generation")
    fingerprint = payload.get("graph_fingerprint")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise AgentResponseError("based_on.generation must be a non-negative integer")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise AgentResponseError("based_on.graph_fingerprint must be a non-empty string")
    return GraphSnapshotRef(graph_slot=_slot_from_dict(payload.get("graph_slot")),
                            generation=generation,
                            graph_fingerprint=fingerprint)


def parse_evaluation_decision(raw_stdout: str,
                              expected_ref: GraphSnapshotRef,
                              stage: str,
                              candidate_generation: Optional[int] = None,
                              candidate_fingerprint: Optional[str] = None) -> EvaluationDecision:
    try:
        payload = json.loads(extract_json_object(raw_stdout))
    except json.JSONDecodeError as exc:
        raise AgentResponseError(f"Evaluator returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AgentResponseError("Evaluation decision must be an object")
    if payload.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise AgentResponseError(f"Unsupported evaluation schema_version {payload.get('schema_version')}")
    based_on = graph_snapshot_from_dict(payload.get("based_on"))
    if based_on != expected_ref:
        raise AgentResponseError(f"Stale evaluation decision: expected {asdict(expected_ref)}, "
                                 f"received {asdict(based_on)}")
    decision = payload.get("decision")
    allowed = {"continue", "finish"} if stage == "accepted_graph" else {"accept", "reject"}
    if decision not in allowed:
        raise AgentResponseError(f"Evaluator decision for stage '{stage}' must be one of {sorted(allowed)}")
    summary = payload.get("summary")
    optimizer_brief = payload.get("optimizer_brief", "")
    if not isinstance(summary, str) or not isinstance(optimizer_brief, str):
        raise AgentResponseError("summary and optimizer_brief must be strings")
    response_generation = payload.get("candidate_generation")
    response_fingerprint = payload.get("candidate_fingerprint")
    if stage == "candidate":
        if response_generation != candidate_generation:
            raise AgentResponseError(f"Evaluator candidate_generation must be {candidate_generation}")
        if response_fingerprint != candidate_fingerprint:
            raise AgentResponseError(f"Evaluator candidate_fingerprint must be {candidate_fingerprint}")
    elif response_generation is not None or response_fingerprint is not None:
        raise AgentResponseError("candidate_generation and candidate_fingerprint must be null when evaluating the "
                                 "accepted graph")
    return EvaluationDecision(schema_version=EVALUATION_SCHEMA_VERSION,
                              based_on=based_on,
                              decision=decision,
                              summary=summary,
                              optimizer_brief=optimizer_brief,
                              candidate_generation=response_generation,
                              candidate_fingerprint=response_fingerprint)


def _profile_payload(profile, bwd: bool) -> Dict[str, Any]:
    times = profile.bwd_time if bwd else profile.fwd_time
    memory = profile.bwd_mem if bwd else profile.fwd_mem
    tensor_sizes = dict(profile.bwd_tensor_sizes if bwd else profile.fwd_tensor_sizes)
    return {
        "total_device_time": sum(row[1] for row in times),
        "peak_memory": max([row[3] for row in memory], default=0),
        "nodes": [{
            "name_hint": name,
            "device_time": device_time,
            "wall_time": wall_time,
            "tensor_size": tensor_sizes.get(name, 0),
        } for name, device_time, wall_time in times],
    }


def _aggregate_candidate_profiles(rank_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [
        result for result in rank_results
        if result.get("success") and isinstance(result.get("metrics"), dict)
    ]
    device_times = [result["metrics"]["device_time"] for result in successful]
    peak_memory = [result["metrics"]["peak_memory"] for result in successful]
    return {
        "rank_results": rank_results,
        "successful_rank_count": len(successful),
        "device_time": {
            "mean": sum(device_times) / len(device_times) if device_times else None,
            "min": min(device_times) if device_times else None,
            "max": max(device_times) if device_times else None,
        },
        "peak_memory": {"max": max(peak_memory) if peak_memory else None},
    }


def serialize_evaluation_context(ctx,
                                 tracker: GraphVersionTracker,
                                 history: List[Dict[str, Any]],
                                 candidate: Optional[Dict[str, Any]] = None) -> str:
    snapshot = tracker.current_ref()
    stage = "candidate" if candidate is not None else "accepted_graph"
    allowed = ["accept", "reject"] if candidate is not None else ["continue", "finish"]
    profile = ctx.profiling_results[ctx.graph_id]
    payload = {
        "role": "deepcompile_evaluator",
        "stage": stage,
        "objective": "Evaluate experimental FX scheduling and graph rewrites using measured execution.",
        "accepted_graph": {
            "snapshot": asdict(snapshot),
            "nodes": normalized_graph(ctx.gm, include_hints=True, runtime_graph_id=ctx.graph_id),
            "profile": _profile_payload(profile, ctx.bwd),
        },
        "candidate": candidate,
        "graph_runtime": {
            "graph_id": dict(RUNTIME_GRAPH_ID),
            "graph_order": normalized_graph_order(ctx.graph_order, ctx.graph_id),
            "direction": "bwd" if ctx.bwd else "fwd",
        },
        "history": history,
        "response_contract": {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "allowed_decisions": allowed,
            "instructions": [
                "Return exactly one JSON object and no prose.",
                "Copy accepted_graph.snapshot into based_on.",
                "At accepted_graph stage, continue requests one optimizer edit and finish ends tuning.",
                "At candidate stage, accept commits the measured candidate and reject keeps the accepted graph.",
                "At candidate stage, copy candidate.generation and candidate.fingerprint into candidate_generation "
                "and candidate_fingerprint. The fingerprint binds both result topology and the exact finalized "
                "data-only edit payload. Set both fields to null at accepted_graph stage.",
                "optimizer_brief may propose any FX topology change, including prefetch or selective-gather style "
                "scheduling, without limiting the optimizer's edit vocabulary.",
            ],
            "fields": {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "based_on": "accepted graph snapshot object",
                "decision": " | ".join(allowed),
                "summary": "evaluation summary",
                "optimizer_brief": "free-form optimizer direction",
                "candidate_generation": "candidate generation or null",
                "candidate_fingerprint": "exact topology-plus-finalized-edit candidate fingerprint or null",
            },
        },
    }
    return json.dumps(payload, indent=2, default=str)


def candidate_evaluation_payload(edit_payload: Dict[str, Any],
                                 candidate_gm: Optional[GraphModule],
                                 rank_results: List[Dict[str, Any]],
                                 runtime_graph_id: int) -> Dict[str, Any]:
    return {
        "generation": edit_payload["generation"],
        "fingerprint": edit_payload["expected_result_fingerprint"],
        "edit": edit_payload,
        "nodes": None if candidate_gm is None else normalized_graph(candidate_gm,
                                                                     include_hints=True,
                                                                     runtime_graph_id=runtime_graph_id),
        "aggregate": _aggregate_candidate_profiles(rank_results),
    }
