# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
from types import SimpleNamespace

import pytest
import torch

from deepspeed.compile import agent_tools
from deepspeed.compile.agent_context import AgentDecision, AgentResponseError, lower_decision_to_plan, parse_agent_response, serialize_agent_context
from deepspeed.compile.profilers import ProfilingResult


class ToyModule(torch.nn.Module):

    def forward(self, x):
        return torch.relu(x + 1)


def build_context():
    gm = torch.fx.symbolic_trace(ToyModule())
    node_names = [node.name for node in gm.graph.nodes]
    profile = ProfilingResult(fwd_graph=gm.graph,
                              fwd_mem=[(name, index * 10, index, index * 10 + 5)
                                       for index, name in enumerate(node_names, start=1)],
                              fwd_time=[(name, index * 3.0, index * 4.0)
                                        for index, name in enumerate(node_names, start=1)],
                              fwd_tensor_sizes=[(name, index * 16) for index, name in enumerate(node_names, start=1)],
                              needs_backward=True)
    return SimpleNamespace(gm=gm,
                           graph_id=17,
                           graph_slot=(2, "fwd"),
                           graph_order=[(17, True)],
                           profiling_results={17: profile},
                           create_inputs_fn=lambda: (),
                           rebuild_structural_baseline_fn=lambda: gm,
                           current_param_manager=SimpleNamespace(params={}, ds_ids={}),
                           all_param_managers={17: SimpleNamespace(params={}, ds_ids={})},
                           mem_budget=0.0,
                           bwd=False,
                           debug_log=False,
                           compile_config=None,
                           warmup_trace=[{
                               "graph_slot": [0, "fwd"],
                               "action": "prefetch"
                           }])


class TestSerializeAgentContext:

    def test_serialize_context_includes_graph_slot_and_trace(self, monkeypatch):
        ctx = build_context()
        monkeypatch.setattr(
            agent_tools, "get_memory_budget_summary", lambda _: {
                "total_memory": 1000,
                "peak_memory": 250,
                "available_memory": 650,
                "memory_margin": 0.1,
            })

        payload = json.loads(
            serialize_agent_context(ctx,
                                    trace_so_far=ctx.warmup_trace,
                                    available_tools=agent_tools.get_available_tools(False, False, set())))

        assert payload["graph_role"]["graph_slot"] == [2, "fwd"]
        assert payload["memory_budget"]["available_memory"] == 650
        assert payload["warmup_trace"][0]["action"] == "prefetch"
        assert payload["available_tools"][0]["name"] == "prefetch"


class TestParseAgentResponse:

    def test_parse_valid_apply_tool(self):
        decision = parse_agent_response('{"decision":"apply_tool","tool_name":"prefetch","reason":"try it"}')
        assert decision.decision == "apply_tool"
        assert decision.tool_name == "prefetch"
        assert decision.reason == "try it"

    def test_parse_valid_finish(self):
        decision = parse_agent_response('{"decision":"finish","reason":"done"}')
        assert decision.decision == "finish"
        assert decision.tool_name is None

    def test_parse_malformed_json(self):
        with pytest.raises(AgentResponseError):
            parse_agent_response("not json")

    def test_parse_unknown_tool(self):
        with pytest.raises(AgentResponseError, match="unknown tool"):
            parse_agent_response('{"decision":"apply_tool","tool_name":"mystery"}')

    def test_parse_missing_fields(self):
        with pytest.raises(AgentResponseError, match="tool_name"):
            parse_agent_response('{"decision":"apply_tool"}')

    def test_parse_json_with_surrounding_text(self):
        decision = parse_agent_response("""
        Here is the decision:
        ```json
        {"decision":"finish","reason":"good enough"}
        ```
        """)
        assert decision.reason == "good enough"


class TestLowerDecisionToPlan:

    def test_lower_prefetch_decision(self, monkeypatch):
        ctx = build_context()
        monkeypatch.setattr(agent_tools, "plan_prefetch", lambda _: {"planned": "prefetch"})

        plan = lower_decision_to_plan(AgentDecision(decision="apply_tool", tool_name="prefetch"), ctx)

        assert plan.tool_name == "prefetch"
        assert plan.payload == {"planned": "prefetch"}

    def test_lower_selective_gather_decision(self, monkeypatch):
        ctx = build_context()
        ctx.bwd = True
        ctx.graph_slot = (2, "bwd")
        monkeypatch.setattr(agent_tools, "plan_selective_gather", lambda _: {"persistent_ds_ids": [1, 2]})

        plan = lower_decision_to_plan(AgentDecision(decision="apply_tool", tool_name="selective_gather"), ctx)

        assert plan.tool_name == "selective_gather"
        assert plan.payload["persistent_ds_ids"] == [1, 2]


class TestToolAvailability:

    def test_tool_availability_forward(self):
        tools = agent_tools.get_available_tools(bwd=False, is_last_backward=False, tools_used=set())
        assert [tool["name"] for tool in tools] == ["prefetch", "finish"]

    def test_tool_availability_last_backward(self):
        tools = agent_tools.get_available_tools(bwd=True, is_last_backward=True, tools_used=set())
        assert [tool["name"] for tool in tools] == ["prefetch", "selective_gather", "finish"]

    def test_tool_availability_prefetch_used(self):
        tools = agent_tools.get_available_tools(bwd=True, is_last_backward=True, tools_used={"prefetch"})
        assert [tool["name"] for tool in tools] == ["selective_gather", "finish"]
