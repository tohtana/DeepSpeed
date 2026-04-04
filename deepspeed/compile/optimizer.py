# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
import gc
import json
import tempfile
from typing import Callable

import torch
from torch.fx import GraphModule

try:
    from torch._subclasses.fake_tensor import unset_fake_temporarily
except ImportError:
    pass

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from .agent_context import AgentResponseError, ToolExecutionPlan, lower_decision_to_plan, parse_agent_response, serialize_agent_context
from .agent_runner import AgentRunner
from .agent_tools import apply_prefetch, apply_selective_gather, get_available_tools, is_last_backward_graph
from .config import CompileConfig
from .graph_param import DSGraphParamManager
from .profilers import ProfilingResult
from .profilers.graph_profile import MemoryProfilingInterpreter


@dataclass
class OptimizationContext:
    gm: GraphModule
    graph_id: int
    graph_slot: tuple[int, str]
    graph_order: list[tuple[int, bool]]
    profiling_results: dict[int, ProfilingResult]
    create_inputs_fn: Callable
    rebuild_structural_baseline_fn: Callable
    current_param_manager: DSGraphParamManager
    all_param_managers: dict[int, DSGraphParamManager]
    mem_budget: float
    bwd: bool
    debug_log: bool
    compile_config: CompileConfig | None
    warmup_trace: list[dict] = field(default_factory=list)


@dataclass
class OptimizationTraceEntry:
    iteration: int
    action: str
    summary: str
    details: dict = field(default_factory=dict)


@dataclass
class OptimizationResult:
    trace: list[OptimizationTraceEntry] = field(default_factory=list)


def _set_time_and_tensor_size(graph_id, graph, mem, bwd, profiling_results):
    node_time = []
    tensor_sizes = []

    for node in graph.nodes:
        node_time.append((node.name, node.meta["device_time"] if "device_time" in node.meta else 0.0,
                          node.meta["wall_time"] if "wall_time" in node.meta else 0.0))
        tensor_sizes.append((node.name, node.meta["tensor_size"] if "tensor_size" in node.meta else 0))

    if bwd:
        profiling_results[graph_id].bwd_graph = graph
        profiling_results[graph_id].bwd_time = node_time
        profiling_results[graph_id].bwd_tensor_sizes = tensor_sizes
        profiling_results[graph_id].bwd_mem = mem
    else:
        profiling_results[graph_id].fwd_graph = graph
        profiling_results[graph_id].fwd_time = node_time
        profiling_results[graph_id].fwd_tensor_sizes = tensor_sizes
        profiling_results[graph_id].fwd_mem = mem


def _profile_graph(gm: GraphModule, ctx: OptimizationContext) -> None:
    mem_prof = MemoryProfilingInterpreter(gm, debug_log=ctx.debug_log)
    mem_prof.run(*ctx.create_inputs_fn())
    mem = [(name, current_alloc, delta, peak) for name, current_alloc, delta, peak in mem_prof.mem_record]
    _set_time_and_tensor_size(ctx.graph_id, gm.graph, mem, ctx.bwd, ctx.profiling_results)


def _cleanup_after_pass() -> None:
    with unset_fake_temporarily():
        get_accelerator().synchronize()
        gc.collect()
        get_accelerator().empty_cache()


class FixedPassOptimizer:
    """Wraps the existing pass loop behind the optimizer interface."""

    def optimize(self, gm: GraphModule, passes: list, ctx: OptimizationContext) -> OptimizationResult:
        trace = []
        _cleanup_after_pass()
        for iteration, opt_pass_fn in enumerate(passes):
            pass_name = getattr(opt_pass_fn, "__name__", str(opt_pass_fn))
            gm_new = opt_pass_fn(gm, ctx.graph_id, ctx.graph_order, ctx.profiling_results, ctx.create_inputs_fn,
                                 ctx.mem_budget, ctx.all_param_managers, ctx.bwd)
            if gm_new is not None:
                gm = gm_new
                ctx.gm = gm
                gm.graph.lint()
                gm.recompile()
                _profile_graph(gm, ctx)

            trace.append(
                OptimizationTraceEntry(iteration=iteration,
                                       action=pass_name,
                                       summary=f"Applied baseline optimization pass '{pass_name}'",
                                       details={
                                           "graph_slot": list(ctx.graph_slot),
                                           "bwd": ctx.bwd
                                       }))
            _cleanup_after_pass()

        return OptimizationResult(trace=trace)


class AgentLoopOptimizer:

    def __init__(self, runner: AgentRunner, compile_config: CompileConfig):
        self.runner = runner
        self.max_iterations = compile_config.agent_max_iterations
        self.timeout_sec = compile_config.agent_timeout_sec
        self.debug_log = compile_config.debug_log

    def _device(self):
        return torch.device(get_accelerator().current_device_name())

    def _broadcast_status_and_plan(self, terminate: bool, plan: ToolExecutionPlan | None) -> ToolExecutionPlan | None:
        if not dist.is_initialized():
            return plan

        try:
            dist.monitored_barrier(timeout=timedelta(seconds=self.timeout_sec + 30))
        except Exception:
            pass

        status = torch.tensor([1 if terminate else 0], dtype=torch.uint8, device=self._device())
        dist.broadcast(status, src=0)
        if status.item() == 1:
            return None

        if dist.get_rank() == 0:
            plan_json = json.dumps(asdict(plan)).encode("utf-8")
            length = torch.tensor([len(plan_json)], dtype=torch.long, device=self._device())
            payload = torch.tensor(list(plan_json), dtype=torch.uint8, device=self._device())
        else:
            length = torch.zeros(1, dtype=torch.long, device=self._device())
            payload = None

        dist.broadcast(length, src=0)
        if dist.get_rank() != 0:
            payload = torch.zeros(length.item(), dtype=torch.uint8, device=self._device())
        dist.broadcast(payload, src=0)

        try:
            dist.monitored_barrier(timeout=timedelta(seconds=self.timeout_sec + 30))
        except Exception:
            pass

        if dist.get_rank() != 0:
            plan_payload = json.loads(bytes(payload.tolist()).decode("utf-8"))
            return ToolExecutionPlan(**plan_payload)
        return plan

    def _rebuild_last_good_graph(self, ctx: OptimizationContext, accepted_plans: list[ToolExecutionPlan]) -> None:
        rebuilt = ctx.rebuild_structural_baseline_fn()
        ctx.gm.graph = rebuilt.graph
        ctx.gm.recompile()
        for plan in accepted_plans:
            if plan.tool_name == "prefetch":
                apply_prefetch(ctx.gm, plan.payload, ctx)
                ctx.gm.graph.lint()
                ctx.gm.recompile()
                _profile_graph(ctx.gm, ctx)

    def optimize(self, gm: GraphModule, ctx: OptimizationContext) -> OptimizationResult:
        trace = []
        tools_used = set()
        accepted_plans = []
        iteration_root = None
        rank0_only = not dist.is_initialized() or dist.get_rank() == 0

        if rank0_only:
            iteration_root = Path(tempfile.mkdtemp(prefix="deepcompile_agent_"))

        for iteration in range(self.max_iterations):
            available_tools = get_available_tools(ctx.bwd, is_last_backward_graph(ctx.graph_id, ctx.graph_order),
                                                  tools_used)

            if rank0_only:
                prompt = serialize_agent_context(ctx,
                                                 trace_so_far=ctx.warmup_trace + [asdict(entry) for entry in trace],
                                                 available_tools=available_tools)
                iteration_dir = iteration_root / f"iter_{iteration}"
                plan = None
                terminate = False
                run_result = None
                try:
                    run_result = self.runner.run(prompt, iteration_dir)
                    if run_result.timed_out:
                        raise AgentResponseError("Agent command timed out")
                    if run_result.returncode != 0:
                        raise AgentResponseError(f"Agent command exited with code {run_result.returncode}")
                    decision = parse_agent_response(run_result.stdout)
                    if decision.decision == "finish":
                        trace.append(
                            OptimizationTraceEntry(iteration=iteration,
                                                   action="finish",
                                                   summary=decision.reason or "Agent chose to keep the current graph",
                                                   details={"graph_slot": list(ctx.graph_slot)}))
                        terminate = True
                    elif decision.tool_name in tools_used:
                        trace.append(
                            OptimizationTraceEntry(
                                iteration=iteration,
                                action="stop",
                                summary=f"Stopping because tool '{decision.tool_name}' already ran on this graph",
                                details={"graph_slot": list(ctx.graph_slot)}))
                        terminate = True
                    else:
                        plan = lower_decision_to_plan(decision, ctx)
                except (AgentResponseError, ValueError) as exc:
                    trace.append(
                        OptimizationTraceEntry(iteration=iteration,
                                               action="stop",
                                               summary=f"Stopping after invalid agent response: {exc}",
                                               details={"graph_slot": list(ctx.graph_slot)}))
                    terminate = True
                finally:
                    keep_dir = self.debug_log or run_result is None or run_result.timed_out or run_result.returncode != 0
                    self.runner.cleanup(iteration_dir, keep=keep_dir or terminate and plan is None)
            else:
                plan = None
                terminate = False

            plan = self._broadcast_status_and_plan(terminate, plan)
            if plan is None:
                break

            try:
                if plan.tool_name == "prefetch":
                    apply_prefetch(ctx.gm, plan.payload, ctx)
                    ctx.gm.graph.lint()
                    ctx.gm.recompile()
                    _profile_graph(ctx.gm, ctx)
                    accepted_plans.append(plan)
                    tools_used.add("prefetch")
                    trace.append(
                        OptimizationTraceEntry(iteration=iteration,
                                               action="prefetch",
                                               summary=plan.reason or "Applied agent-selected prefetch rewrite",
                                               details={"graph_slot": list(ctx.graph_slot)}))
                elif plan.tool_name == "selective_gather":
                    newly_marked = apply_selective_gather(plan.payload, ctx)
                    trace.append(
                        OptimizationTraceEntry(iteration=iteration,
                                               action="selective_gather",
                                               summary=plan.reason or "Applied agent-selected selective gather",
                                               details={
                                                   "graph_slot": list(ctx.graph_slot),
                                                   "persistent_ds_ids": newly_marked,
                                               }))
                    break
                else:
                    raise ValueError(f"Unsupported tool plan '{plan.tool_name}'")
            except Exception as exc:
                if accepted_plans:
                    self._rebuild_last_good_graph(ctx, accepted_plans)
                trace.append(
                    OptimizationTraceEntry(iteration=iteration,
                                           action="stop",
                                           summary=f"Stopping after tool application failure: {exc}",
                                           details={"graph_slot": list(ctx.graph_slot)}))
                break

        return OptimizationResult(trace=trace)
