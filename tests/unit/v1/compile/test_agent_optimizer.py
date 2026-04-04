# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

import torch

from deepspeed.compile.agent_context import ToolExecutionPlan
from deepspeed.compile.config import CompileConfig
from deepspeed.compile.optimizer import AgentLoopOptimizer, OptimizationContext
from deepspeed.compile.profilers import ProfilingResult


class ToyModule(torch.nn.Module):

    def forward(self, x):
        return x + 1


class FakeRunner:

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.cleaned = []

    def run(self, prompt, iteration_dir):
        output = self.outputs.pop(0)
        return SimpleNamespace(stdout=output,
                               stderr="",
                               returncode=0,
                               timed_out=False,
                               prompt_path=str(iteration_dir / "prompt.txt"),
                               stdout_path=str(iteration_dir / "stdout.txt"),
                               stderr_path=str(iteration_dir / "stderr.txt"))

    def cleanup(self, iteration_dir, keep):
        self.cleaned.append((str(iteration_dir), keep))


def build_context():
    gm = torch.fx.symbolic_trace(ToyModule())
    profile = ProfilingResult(fwd_graph=gm.graph,
                              fwd_mem=[(node.name, 1, 0, 1) for node in gm.graph.nodes],
                              fwd_time=[(node.name, 1.0, 1.0) for node in gm.graph.nodes],
                              fwd_tensor_sizes=[(node.name, 4) for node in gm.graph.nodes])
    return gm, OptimizationContext(gm=gm,
                                   graph_id=11,
                                   graph_slot=(0, "fwd"),
                                   graph_order=[(11, True)],
                                   profiling_results={11: profile},
                                   create_inputs_fn=lambda: (),
                                   rebuild_structural_baseline_fn=lambda: gm,
                                   current_param_manager=SimpleNamespace(params={}, ds_ids={}),
                                   all_param_managers={11: SimpleNamespace(params={}, ds_ids={})},
                                   mem_budget=0.0,
                                   bwd=False,
                                   debug_log=False,
                                   compile_config=CompileConfig(zero3_tuning_strategy="agent",
                                                                agent_command=["/bin/bash", "-lc", "cat"],
                                                                agent_max_iterations=3,
                                                                agent_timeout_sec=5),
                                   warmup_trace=[])


class TestAgentLoopOptimizer:

    def test_broadcast_plan_skips_single_rank(self, monkeypatch):
        _, ctx = build_context()
        optimizer = AgentLoopOptimizer(FakeRunner([]), ctx.compile_config)
        plan = ToolExecutionPlan(tool_name="prefetch", payload={"sequence": []}, reason="noop")
        broadcast_called = False

        monkeypatch.setattr("deepspeed.compile.optimizer.dist.is_initialized", lambda: True)
        monkeypatch.setattr("deepspeed.compile.optimizer.dist.get_world_size", lambda: 1)

        def fail_broadcast(*args, **kwargs):
            nonlocal broadcast_called
            broadcast_called = True
            raise AssertionError("broadcast should not run for single-rank agent mode")

        monkeypatch.setattr("deepspeed.compile.optimizer.dist.broadcast", fail_broadcast)

        result = optimizer._broadcast_status_and_plan(False, plan)

        assert result is plan
        assert not broadcast_called

    def test_agent_loop_finish_immediately(self, monkeypatch):
        _, ctx = build_context()
        runner = FakeRunner(['{"decision":"finish","reason":"done"}'])
        monkeypatch.setattr("deepspeed.compile.optimizer.serialize_agent_context", lambda *args, **kwargs: "prompt")

        result = AgentLoopOptimizer(runner, ctx.compile_config).optimize(ctx.gm, ctx)

        assert [entry.action for entry in result.trace] == ["finish"]

    def test_agent_loop_prefetch_then_finish(self, monkeypatch):
        _, ctx = build_context()
        runner = FakeRunner([
            '{"decision":"apply_tool","tool_name":"prefetch","reason":"overlap"}',
            '{"decision":"finish","reason":"stop"}',
        ])
        calls = []

        monkeypatch.setattr("deepspeed.compile.optimizer.serialize_agent_context", lambda *args, **kwargs: "prompt")
        monkeypatch.setattr("deepspeed.compile.optimizer.lower_decision_to_plan",
                            lambda decision, _ctx: ToolExecutionPlan(tool_name=decision.tool_name, payload={}))
        monkeypatch.setattr("deepspeed.compile.optimizer.apply_prefetch",
                            lambda gm, payload, _ctx: calls.append(payload))
        monkeypatch.setattr("deepspeed.compile.optimizer._profile_graph", lambda gm, _ctx: None)

        result = AgentLoopOptimizer(runner, ctx.compile_config).optimize(ctx.gm, ctx)

        assert calls == [{}]
        assert [entry.action for entry in result.trace] == ["prefetch", "finish"]

    def test_agent_loop_invalid_response(self, monkeypatch):
        _, ctx = build_context()
        runner = FakeRunner(["not json"])
        monkeypatch.setattr("deepspeed.compile.optimizer.serialize_agent_context", lambda *args, **kwargs: "prompt")

        result = AgentLoopOptimizer(runner, ctx.compile_config).optimize(ctx.gm, ctx)

        assert result.trace[0].action == "stop"

    def test_agent_loop_selective_gather_terminal(self, monkeypatch):
        _, ctx = build_context()
        ctx.bwd = True
        ctx.graph_slot = (0, "bwd")
        runner = FakeRunner(['{"decision":"apply_tool","tool_name":"selective_gather","reason":"persist"}'])

        monkeypatch.setattr("deepspeed.compile.optimizer.serialize_agent_context", lambda *args, **kwargs: "prompt")
        monkeypatch.setattr(
            "deepspeed.compile.optimizer.lower_decision_to_plan",
            lambda decision, _ctx: ToolExecutionPlan(tool_name=decision.tool_name,
                                                     payload={"persistent_ds_ids": [1, 2]}))
        monkeypatch.setattr("deepspeed.compile.optimizer.apply_selective_gather", lambda payload, _ctx: [1, 2])

        result = AgentLoopOptimizer(runner, ctx.compile_config).optimize(ctx.gm, ctx)

        assert [entry.action for entry in result.trace] == ["selective_gather"]
        assert result.trace[0].details["persistent_ds_ids"] == [1, 2]

    def test_agent_loop_prefetch_single_use(self, monkeypatch):
        _, ctx = build_context()
        runner = FakeRunner([
            '{"decision":"apply_tool","tool_name":"prefetch","reason":"first"}',
            '{"decision":"apply_tool","tool_name":"prefetch","reason":"again"}',
        ])

        monkeypatch.setattr("deepspeed.compile.optimizer.serialize_agent_context", lambda *args, **kwargs: "prompt")
        monkeypatch.setattr("deepspeed.compile.optimizer.lower_decision_to_plan",
                            lambda decision, _ctx: ToolExecutionPlan(tool_name=decision.tool_name, payload={}))
        monkeypatch.setattr("deepspeed.compile.optimizer.apply_prefetch", lambda gm, payload, _ctx: gm)
        monkeypatch.setattr("deepspeed.compile.optimizer._profile_graph", lambda gm, _ctx: None)

        result = AgentLoopOptimizer(runner, ctx.compile_config).optimize(ctx.gm, ctx)

        assert [entry.action for entry in result.trace] == ["prefetch", "stop"]

    def test_agent_loop_max_iterations(self, monkeypatch):
        _, ctx = build_context()
        ctx.compile_config.agent_max_iterations = 1
        runner = FakeRunner(['{"decision":"apply_tool","tool_name":"prefetch","reason":"once"}'])

        monkeypatch.setattr("deepspeed.compile.optimizer.serialize_agent_context", lambda *args, **kwargs: "prompt")
        monkeypatch.setattr("deepspeed.compile.optimizer.lower_decision_to_plan",
                            lambda decision, _ctx: ToolExecutionPlan(tool_name=decision.tool_name, payload={}))
        monkeypatch.setattr("deepspeed.compile.optimizer.apply_prefetch", lambda gm, payload, _ctx: gm)
        monkeypatch.setattr("deepspeed.compile.optimizer._profile_graph", lambda gm, _ctx: None)

        result = AgentLoopOptimizer(runner, ctx.compile_config).optimize(ctx.gm, ctx)

        assert [entry.action for entry in result.trace] == ["prefetch"]
