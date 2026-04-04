# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict
from typing import TYPE_CHECKING

import torch
from torch.fx import Graph, GraphModule, Node

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from .passes import prefetch as prefetch_pass
from .util import get_deepcompile_handle

if TYPE_CHECKING:
    from .optimizer import OptimizationContext

TOOL_SCHEMAS = {
    "prefetch": {
        "name": "prefetch",
        "description": "Reorder and fuse all-gather operations to improve overlap between communication and compute.",
        "kwargs": {},
        "constraints": "Single-use per graph in v1. Produces a new profile-visible graph.",
    },
    "selective_gather": {
        "name": "selective_gather",
        "description": "Mark parameters as persistent when memory headroom allows it.",
        "kwargs": {},
        "constraints": "Only valid on the last backward graph. Terminal action in v1.",
    },
    "finish": {
        "name": "finish",
        "description": "Stop tuning and keep the current graph.",
        "kwargs": {},
    },
}


def is_last_backward_graph(graph_id: int, graph_order: list[tuple[int, bool]]) -> bool:
    last_backward_graph_id = None
    for g_id, needs_bwd in graph_order:
        if needs_bwd:
            last_backward_graph_id = g_id
            break
    return last_backward_graph_id is not None and graph_id == last_backward_graph_id


def get_memory_budget_summary(profiling_results, synchronize_ranks: bool = True) -> dict:
    peak_mem = 0
    for profile in profiling_results.values():
        if profile.fwd_mem:
            peak_mem = max(peak_mem, max(mem[3] for mem in profile.fwd_mem))
        if profile.bwd_mem:
            peak_mem = max(peak_mem, max(mem[3] for mem in profile.bwd_mem))

    total_mem = get_accelerator().total_memory()
    if synchronize_ranks and dist.is_initialized():
        vals_to_bcast = torch.tensor([total_mem], device=torch.device(get_accelerator().current_device()))
        dist.all_reduce(vals_to_bcast, dist.ReduceOp.MIN)
        total_mem = vals_to_bcast[0].item()

    mem_margin = 0.1
    return {
        "total_memory": total_mem,
        "peak_memory": peak_mem,
        "available_memory": total_mem * (1 - mem_margin) - peak_mem,
        "memory_margin": mem_margin,
    }


def get_available_tools(bwd: bool, is_last_backward: bool, tools_used: set[str]) -> list[dict]:
    tool_names = ["finish"]
    if "prefetch" not in tools_used:
        tool_names.insert(0, "prefetch")
    if bwd and is_last_backward:
        tool_names.insert(-1, "selective_gather")
    return [dict(TOOL_SCHEMAS[name]) for name in tool_names]


def _get_profile_maps(gm: GraphModule, graph_id: int, profiling_results, bwd: bool):
    mem = profiling_results[graph_id].bwd_mem if bwd else profiling_results[graph_id].fwd_mem
    op_time = profiling_results[graph_id].bwd_time if bwd else profiling_results[graph_id].fwd_time
    tensor_sizes = profiling_results[graph_id].bwd_tensor_sizes if bwd else profiling_results[graph_id].fwd_tensor_sizes

    mem_dict = {name: (alloc_mem, peak) for name, alloc_mem, _, peak in mem}
    time_dict = {name: (device_time, wall_time) for name, device_time, wall_time in op_time}
    tensor_size_dict = {name: size for name, size in tensor_sizes}

    prev_mem = 0
    prev_peak = 0
    for node in gm.graph.nodes:
        if node.name in mem_dict:
            prev_mem = mem_dict[node.name][0]
            prev_peak = mem_dict[node.name][1]
        else:
            mem_dict[node.name] = (prev_mem, prev_peak)

    return mem_dict, time_dict, tensor_size_dict


def plan_prefetch(ctx: "OptimizationContext") -> dict:
    max_mem = get_accelerator().total_memory() * (1 - prefetch_pass.MARGIN)
    if dist.is_initialized():
        vals_to_bcast = torch.tensor([max_mem], device=torch.device(get_accelerator().current_device()))
        dist.all_reduce(vals_to_bcast, dist.ReduceOp.MIN)
        max_mem = vals_to_bcast[0].item()

    gm = ctx.gm
    mem_dict, _, tensor_size_dict = _get_profile_maps(gm, ctx.graph_id, ctx.profiling_results, ctx.bwd)
    comm_predictor = prefetch_pass.create_predictor()

    order_rev = list(reversed(gm.graph.nodes))
    new_order_rev = []
    prefetch_ags = []
    prefetch_ag_groups = []
    ag_tensor_size_sum = 0
    for i, node in enumerate(order_rev):
        if node.op != "placeholder":
            next_node = order_rev[i + 1]
            _, next_peak = mem_dict[next_node.name]

            while next_peak + ag_tensor_size_sum > max_mem or ag_tensor_size_sum > prefetch_pass.MAX_BUFFERED_SIZE:
                if prefetch_ag_groups:
                    fused_ag_nodes = prefetch_ag_groups.pop(0)
                    total_ag_tensor_size = sum([tensor_size_dict[ag_node.name] for ag_node in fused_ag_nodes])
                    ag_tensor_size_sum -= total_ag_tensor_size
                    new_order_rev.append(fused_ag_nodes)
                elif prefetch_ags:
                    prefetch_ag_groups.append(prefetch_ags)
                    prefetch_ags = []
                else:
                    break

            if node.target == torch.ops.dc.allgather_param.default:
                current_ag_size = sum([tensor_size_dict[ag_node.name] for ag_node in prefetch_ags])
                pred_time_current = comm_predictor(current_ag_size)
                pred_time_next = comm_predictor(tensor_size_dict[node.name])
                pred_time_fused = comm_predictor(current_ag_size + tensor_size_dict[node.name])

                do_fuse = max(pred_time_current, pred_time_next) * 1.2 > pred_time_fused and (
                    current_ag_size + tensor_size_dict[node.name]) < prefetch_pass.MAX_FUSE_SIZE

                if prefetch_ags and not do_fuse:
                    prefetch_ag_groups.append(prefetch_ags)
                    prefetch_ags = []

                prefetch_ags.append(node)
                ag_tensor_size_sum += tensor_size_dict[node.name]

        new_order_rev.append(node)

        if (node.op != "placeholder"
                and node.target != torch.ops.dc.reload_parameter) and order_rev[i + 1].op == "placeholder":
            for ag_group in prefetch_ag_groups:
                new_order_rev.append(ag_group)
                ag_tensor_size_sum -= sum([tensor_size_dict[ag_node.name] for ag_node in ag_group])
            if prefetch_ags:
                new_order_rev.append(prefetch_ags)
                ag_tensor_size_sum -= sum([tensor_size_dict[ag_node.name] for ag_node in prefetch_ags])

        assert ag_tensor_size_sum >= 0

    sequence = []
    for item in reversed(new_order_rev):
        if isinstance(item, Node):
            sequence.append({"kind": "node", "name": item.name})
        else:
            sequence.append({
                "kind": "prefetch",
                "param_node_names": [ag_node.args[0].name for ag_node in item],
                "ds_ids": [prefetch_pass.get_ds_id(ag_node) for ag_node in item],
            })

    return {"sequence": sequence}


def apply_prefetch(gm: GraphModule, plan_payload: dict, ctx: "OptimizationContext") -> GraphModule:
    node_by_name = {node.name: node for node in gm.graph.nodes}
    new_graph = Graph()
    env = {}

    for item in plan_payload["sequence"]:
        if item["kind"] == "node":
            node = node_by_name[item["name"]]
            env[node.name] = new_graph.node_copy(node, lambda n: env[n.name])
            continue

        param_nodes = [env[param_node_name] for param_node_name in item["param_node_names"]]
        new_graph.call_function(torch.ops.dc.prefetch_params_fused.default,
                                args=(ctx.graph_id, param_nodes, item["ds_ids"]))

    new_graph.lint()
    gm.graph = new_graph
    return gm


def plan_selective_gather(ctx: "OptimizationContext") -> dict:
    if not ctx.bwd or not is_last_backward_graph(ctx.graph_id, ctx.graph_order):
        raise ValueError("selective_gather is only valid on the last backward graph")

    peak_mem = 0
    for _, profile in ctx.profiling_results.items():
        if profile.fwd_mem:
            peak_mem = max(peak_mem, max(m[3] for m in profile.fwd_mem))
        if profile.bwd_mem:
            peak_mem = max(peak_mem, max(m[3] for m in profile.bwd_mem))

    persistent_ds_ids = set()
    for _, param_mgr in ctx.all_param_managers.items():
        for name, ds_param in param_mgr.params.items():
            if ds_param.param.ds_persist:
                persistent_ds_ids.add(param_mgr.ds_ids[name])

    ds_id_to_size = {}
    ds_id_to_time = defaultdict(float)
    for graph_id, param_mgr in ctx.all_param_managers.items():
        for param_name, ds_param in param_mgr.params.items():
            ds_id = param_mgr.ds_ids[param_name]
            ds_id_to_size[ds_id] = ds_param.numel * ds_param.dtype.itemsize

        profile = ctx.profiling_results[graph_id]
        for graph in [profile.fwd_graph, profile.bwd_graph]:
            if graph is None:
                continue
            for node in graph.nodes:
                if node.target == torch.ops.dc.allgather_param.default:
                    ds_id = node.args[2]
                    ds_id_to_size[ds_id] = node.meta["tensor_size"]
                    ds_id_to_time[ds_id] += node.meta["device_time"]

    total_mem = get_memory_budget_summary(ctx.profiling_results)["total_memory"]
    available_mem = total_mem * 0.9 - peak_mem

    sorted_ds_ids = [ds_id for ds_id in ds_id_to_size if ds_id not in persistent_ds_ids]
    sorted_ds_ids.sort(key=lambda ds_id: ds_id_to_time[ds_id] / ds_id_to_size[ds_id], reverse=True)

    chosen_ds_ids = []
    persistent_mem = 0
    for ds_id in sorted_ds_ids:
        size = ds_id_to_size[ds_id]
        if persistent_mem + size > available_mem:
            break
        persistent_mem += size
        chosen_ds_ids.append(ds_id)

    return {
        "persistent_ds_ids": chosen_ds_ids,
        "peak_memory": peak_mem,
        "available_memory": available_mem,
    }


def apply_selective_gather(plan_payload: dict, ctx: "OptimizationContext") -> list[int]:
    ds_id_to_param = {}
    for _, param_mgr in ctx.all_param_managers.items():
        for name, ds_param in param_mgr.params.items():
            ds_id_to_param[param_mgr.ds_ids[name]] = ds_param.param

    nz3 = get_deepcompile_handle()
    newly_marked = []
    for ds_id in plan_payload["persistent_ds_ids"]:
        param = ds_id_to_param.get(ds_id)
        if param is not None and getattr(param, "ds_persist", False):
            continue

        nz3.set_persistent(ds_id)
        if param is not None:
            param.ds_persist = True
        newly_marked.append(ds_id)

    return newly_marked
