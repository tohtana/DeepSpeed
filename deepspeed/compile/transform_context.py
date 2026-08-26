# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Dict, List

from .evaluation_context import AgentResponseError, EvaluationDecision, GraphVersionTracker, extract_json_object
from .graph_edit import (GraphEditPayload, RUNTIME_GRAPH_ID, SCHEMA_VERSION, normalized_graph,
                         normalized_graph_order)


def serialize_transform_context(ctx,
                                evaluation: EvaluationDecision,
                                tracker: GraphVersionTracker,
                                history: List[Dict[str, Any]],
                                mechanical_feedback: List[str]) -> str:
    snapshot = tracker.current_ref()
    payload = {
        "role": "deepcompile_graph_optimizer",
        "objective": "Produce one complete generic FX graph edit for the evaluator's measured optimization loop.",
        "graph_snapshot": asdict(snapshot),
        "graph_runtime": {
            "graph_id": dict(RUNTIME_GRAPH_ID),
            "graph_slot": list(ctx.graph_slot),
            "graph_order": normalized_graph_order(ctx.graph_order, ctx.graph_id),
            "direction": "bwd" if ctx.bwd else "fwd",
        },
        "graph_nodes": normalized_graph(ctx.gm, include_hints=True, runtime_graph_id=ctx.graph_id),
        "evaluator": evaluation.to_dict(),
        "history": history,
        "mechanical_feedback": mechanical_feedback,
        "edit_contract": {
            "schema_version": SCHEMA_VERSION,
            "identity": {
                "generation": snapshot.generation + 1,
                "graph_slot": list(ctx.graph_slot),
                "base_fingerprint": snapshot.graph_fingerprint,
                "expected_result_fingerprint": None,
            },
            "operations": {
                "create_node": {
                    "fields": [
                        "op=create_node", "id", "node_op", "target", "args", "kwargs", "name_hint(optional)",
                        "copy_meta_from(optional)", "meta(optional recursively JSON/data-only patches)"
                    ],
                    "node_ops": [
                        "placeholder", "get_attr", "call_function", "call_method", "call_module", "output"
                    ],
                    "targets": "call_function uses an importable symbolic path such as torch.neg, operator.getitem, "
                    "or torch.ops.dc.prefetch_params_fused.default; other node kinds use their normal FX string "
                    "target",
                },
                "set_args_kwargs": "Set complete args and/or kwargs for any existing or new node. 'rewire' is an "
                "alias.",
                "delete_node": "Delete a node after all uses have been rewired.",
                "patch_meta": "Apply recursively JSON/data-only metadata patches using scalars, lists, and "
                "string-keyed objects; new nodes may instead copy metadata from a local node.",
                "reorder": "The final operation, naming every final node ID exactly once in target topological order.",
            },
            "argument_encoding": {
                "node_reference": {"node": "base:topological_position or new ID"},
                "json_scalars_and_lists": "encoded directly and recursively",
                "tuple": {"tuple": []},
                "dictionary": {"dict": [["encoded key", "encoded value"]]},
                "slice": {"slice": ["start", "stop", "step"]},
                "ellipsis": {"ellipsis": True},
                "torch_dtype": {"torch_dtype": "float32"},
                "torch_device": {"torch_device": {"type": "cuda", "index": "current"}},
                "python_symbol": {"python_symbol": "importable.module.symbol"},
                "runtime_graph_id": dict(RUNTIME_GRAPH_ID),
            },
            "mechanical_rules": [
                "Return exactly one JSON object and no prose.",
                "Use stable base IDs from graph_nodes; name_hint is never node identity.",
                "Leave expected_result_fingerprint null; rank 0 computes it from the replayed result topology and "
                "the canonical finalized data-only edit payload.",
                "Create operations must precede references to their new IDs.",
                "Rewire uses before deleting nodes, then finish with the complete topological reorder.",
                "All callable/module/get_attr targets must resolve locally and graph.lint/recompile must succeed.",
                "Use the runtime_graph_id encoding wherever a new or rewired operation needs this rank's graph ID.",
                "There are no semantic operator allowlists or bans: collectives, compute, replacements, and arbitrary "
                "valid FX reorderings use the same generic operations.",
                "Existing local nodes retain rank-local callables, parameters, modules, metadata, and get_attr "
                "values.",
                "The edit must remain callable with the existing positional AOT inputs and preserve the output "
                "pytree/container and leaf-kind ABI; placeholder names may differ and outputs may be rewired.",
                "Do not serialize tensors or FakeTensors. Copy local metadata and let profiling refresh "
                "timing/memory.",
            ],
            "required_fields": [
                "schema_version", "generation", "graph_slot", "base_fingerprint",
                "expected_result_fingerprint", "reason", "operations"
            ],
        },
    }
    return json.dumps(payload, indent=2, default=str)


def parse_optimizer_edit(raw_stdout: str, tracker: GraphVersionTracker) -> GraphEditPayload:
    try:
        data = json.loads(extract_json_object(raw_stdout))
    except json.JSONDecodeError as exc:
        raise AgentResponseError(f"Optimizer returned invalid JSON: {exc}") from exc
    try:
        payload = GraphEditPayload.from_dict(data)
    except ValueError as exc:
        raise AgentResponseError(str(exc)) from exc
    snapshot = tracker.current_ref()
    expected_slot = (snapshot.graph_slot.index, snapshot.graph_slot.direction)
    if payload.graph_slot != expected_slot:
        raise AgentResponseError(f"Optimizer edit targets graph slot {payload.graph_slot}, expected {expected_slot}")
    if payload.generation != snapshot.generation + 1:
        raise AgentResponseError(f"Optimizer edit generation {payload.generation} does not follow accepted generation "
                                 f"{snapshot.generation}")
    if payload.base_fingerprint != snapshot.graph_fingerprint:
        raise AgentResponseError(f"Optimizer edit base {payload.base_fingerprint} does not match accepted graph "
                                 f"{snapshot.graph_fingerprint}")
    if payload.expected_result_fingerprint is not None:
        raise AgentResponseError("Optimizer must leave expected_result_fingerprint null for rank-zero finalization")
    return payload
