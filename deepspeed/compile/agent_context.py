# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass, field
import json

from . import agent_tools


class AgentResponseError(ValueError):
    pass


@dataclass
class AgentDecision:
    decision: str
    tool_name: str | None = None
    tool_kwargs: dict[str, object] = field(default_factory=dict)
    reason: str = ""


@dataclass
class ToolExecutionPlan:
    tool_name: str
    payload: dict[str, object]
    reason: str = ""


def _extract_json_object(raw_stdout: str) -> str:
    text = raw_stdout.strip()
    if not text:
        raise AgentResponseError("Agent returned empty output")

    if "```" in text:
        blocks = text.split("```")
        for block in blocks:
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


def _summarize_profile(profile, bwd: bool) -> dict:
    mem = profile.bwd_mem if bwd else profile.fwd_mem
    time = profile.bwd_time if bwd else profile.fwd_time
    tensor_sizes = profile.bwd_tensor_sizes if bwd else profile.fwd_tensor_sizes
    peak_mem = max([entry[3] for entry in mem], default=0)
    total_device_time = sum([entry[1] for entry in time])
    top_nodes = sorted(time, key=lambda entry: entry[1], reverse=True)[:5]
    return {
        "peak_memory":
        peak_mem,
        "total_device_time":
        total_device_time,
        "top_nodes": [{
            "name": name,
            "device_time": device_time,
            "wall_time": wall_time,
            "tensor_size": dict(tensor_sizes).get(name, 0),
        } for name, device_time, wall_time in top_nodes],
    }


def _summarize_graph(graph) -> dict:
    nodes = list(graph.nodes)
    op_histogram = {}
    call_nodes = []
    placeholders = []

    for node in nodes:
        op_histogram[node.op] = op_histogram.get(node.op, 0) + 1
        if node.op == "placeholder" and len(placeholders) < 16:
            placeholders.append(node.name)
        if node.op.startswith("call") and len(call_nodes) < 32:
            target = getattr(node.target, "__name__", None) or str(node.target)
            call_nodes.append({
                "name": node.name,
                "op": node.op,
                "target": target,
            })

    return {
        "node_count": len(nodes),
        "op_histogram": op_histogram,
        "placeholders": placeholders,
        "call_nodes": call_nodes,
    }


def serialize_agent_context(ctx, trace_so_far: list[dict], available_tools: list[dict]) -> str:
    graph_role = {
        "graph_slot": list(ctx.graph_slot),
        "graph_id": ctx.graph_id,
        "graph_order": ctx.graph_order,
        "direction": "bwd" if ctx.bwd else "fwd",
        "is_last_backward": agent_tools.is_last_backward_graph(ctx.graph_id, ctx.graph_order),
    }

    current_profile = _summarize_profile(ctx.profiling_results[ctx.graph_id], ctx.bwd)
    cross_graph_summary = []
    for graph_id, profile in ctx.profiling_results.items():
        cross_graph_summary.append({
            "graph_id": graph_id,
            "needs_backward": profile.needs_backward,
            "fwd_peak_memory": max([entry[3] for entry in profile.fwd_mem], default=0),
            "bwd_peak_memory": max([entry[3] for entry in profile.bwd_mem], default=0),
        })

    payload = {
        "objective":
        "Modify the ZeRO-3 graph to maximize throughput for distributed training without exceeding memory limits.",
        "constraints": [
            "Preserve graph correctness.",
            "Use only the provided tools.",
            "Stop when further changes are unlikely to help.",
        ],
        "graph_role":
        graph_role,
        "memory_budget":
        agent_tools.get_memory_budget_summary(ctx.profiling_results, synchronize_ranks=False),
        "current_profile":
        current_profile,
        "cross_graph_summary":
        cross_graph_summary,
        "warmup_trace":
        trace_so_far,
        "available_tools":
        available_tools,
        "graph_summary":
        _summarize_graph(ctx.gm.graph),
    }
    return json.dumps(payload, indent=2, default=str)


def parse_agent_response(raw_stdout: str) -> AgentDecision:
    try:
        payload = json.loads(_extract_json_object(raw_stdout))
    except json.JSONDecodeError as exc:
        raise AgentResponseError(f"Agent returned invalid JSON: {exc}") from exc

    decision = payload.get("decision")
    if decision not in {"apply_tool", "finish"}:
        raise AgentResponseError("Agent response must set decision to 'apply_tool' or 'finish'")

    tool_name = payload.get("tool_name")
    if decision == "apply_tool":
        if not tool_name:
            raise AgentResponseError("Agent response is missing tool_name for apply_tool")
        if tool_name not in {"prefetch", "selective_gather"}:
            raise AgentResponseError(f"Agent requested unknown tool '{tool_name}'")

    tool_kwargs = payload.get("tool_kwargs", {})
    if tool_kwargs is None:
        tool_kwargs = {}
    if not isinstance(tool_kwargs, dict):
        raise AgentResponseError("Agent response field tool_kwargs must be an object")

    reason = payload.get("reason", "")
    if reason is None:
        reason = ""

    return AgentDecision(decision=decision, tool_name=tool_name, tool_kwargs=tool_kwargs, reason=reason)


def lower_decision_to_plan(decision: AgentDecision, ctx) -> ToolExecutionPlan:
    if decision.decision != "apply_tool" or decision.tool_name is None:
        raise AgentResponseError("Only apply_tool decisions can be lowered into execution plans")

    if decision.tool_name == "prefetch":
        payload = agent_tools.plan_prefetch(ctx)
    elif decision.tool_name == "selective_gather":
        payload = agent_tools.plan_selective_gather(ctx)
    else:
        raise AgentResponseError(f"Unsupported tool '{decision.tool_name}'")

    return ToolExecutionPlan(tool_name=decision.tool_name, payload=payload, reason=decision.reason)
