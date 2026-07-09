# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import gc
import os
from dataclasses import replace
from typing import List, Dict, Tuple
import _operator

import torch
from torch.fx import Graph, Node, GraphModule

from ..util import get_input_nodes, get_param_nodes, get_index_by_graph_id, get_deepcompile_handle, get_real_uses, is_cast_op
from ..fx import (add_postprocess, _make_node_meta, get_output_node, move_primals_to_head, add_end_backward,
                  replace_reduce_outputs_with_none, should_release_reduce_buckets)
from ..profilers.graph_profile import ProfilingInterpreter, is_profile_incomplete
from ..list_schedule import (SCHEDULER_BUDGET_DIAGNOSTICS_ATTR, SchedulerMemoryBudget, allgather_allocation_bytes,
                             fast_free_schedule, max_possible_gathered_bytes)

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

NAME = "zero3_compile"
SCHEDULER_DEBUG_ENV = "DEEPSPEED_COMPILE_SCHEDULER_BUDGET_DEBUG"
SCHEDULER_DEBUG_ENV_LEGACY = "DEEPSPEED_DEEPCOMPILE_SCHEDULER_DEBUG"


def _reduce_int(value: int, op):
    if not dist.is_initialized():
        return int(value)

    value_tensor = torch.tensor([int(value)],
                                device=torch.device(get_accelerator().current_device()),
                                dtype=torch.int64)
    dist.all_reduce(value_tensor, op)
    return int(value_tensor.item())


def _rank_min_total_memory():
    return _reduce_int(get_accelerator().total_memory(), dist.ReduceOp.MIN)


def _rank_min_available_memory():
    return _reduce_int(get_accelerator().available_memory(), dist.ReduceOp.MIN)


def _world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def _sync_profile_complete(profile_complete: bool):
    if not dist.is_initialized():
        return profile_complete

    complete = torch.tensor([1 if profile_complete else 0],
                            device=torch.device(get_accelerator().current_device()),
                            dtype=torch.int)
    dist.all_reduce(complete, dist.ReduceOp.MIN)
    return bool(complete.item())


def _operator_profile_complete(graph: Graph):
    return not is_profile_incomplete(graph) and all("max_mem" in node.meta for node in graph.nodes)


def _is_gather_lifetime_node(node: Node):
    return node.target in (torch.ops.dc.allgather_param.default, torch.ops.dc.wait_allgather.default,
                           torch.ops.dc.release_param.default, torch.ops.dc.reduce_grad.default)


def _operator_profile_has_observed_non_gathered_peak(graph: Graph):
    return any(not _is_gather_lifetime_node(node) and int(node.meta.get("max_mem", 0) or 0) > 0
               for node in graph.nodes)


def _rank_max_operator_profiled_non_gathered_peak(graph: Graph):
    peak = 0
    for node in graph.nodes:
        if _is_gather_lifetime_node(node):
            continue
        peak = max(peak, int(node.meta.get("max_mem", 0) or 0))
    return _reduce_int(int(get_accelerator().memory_allocated()) + peak, dist.ReduceOp.MAX)


def _build_scheduler_budget_from_operator_profile(graph: Graph, output_size: int = 0):
    if not _operator_profile_complete(graph):
        return None

    return SchedulerMemoryBudget.from_profiled_non_gathered_peak(_rank_min_total_memory(),
                                                                 _rank_max_operator_profiled_non_gathered_peak(graph),
                                                                 output_size)


def _build_scheduler_budget_from_partial_operator_profile(graph: Graph, output_size: int = 0):
    if not _operator_profile_has_observed_non_gathered_peak(graph):
        return None

    return SchedulerMemoryBudget.from_profiled_non_gathered_peak(_rank_min_total_memory(),
                                                                 _rank_max_operator_profiled_non_gathered_peak(graph),
                                                                 output_size)


def _max_single_allgather_allocation_bytes(graph: Graph):
    return max((int(node.meta.get("allgather_allocation_bytes", node.meta.get("tensor_size", 0)) or 0)
                for node in graph.nodes if node.target == torch.ops.dc.allgather_param.default),
               default=0)


def _cap_incomplete_profile_budget(graph: Graph, scheduler_budget):
    if scheduler_budget is None:
        return None
    max_single_allgather_bytes = _max_single_allgather_allocation_bytes(graph)
    if max_single_allgather_bytes <= 0 or scheduler_budget.max_gathered_bytes <= max_single_allgather_bytes:
        return scheduler_budget
    return replace(scheduler_budget,
                   max_gathered_bytes=max_single_allgather_bytes,
                   source=f"{scheduler_budget.source}_single_allgather_cap")


def _scheduler_debug_enabled():
    return any(
        os.environ.get(env_name, "").lower() not in ("", "0", "false", "no")
        for env_name in (SCHEDULER_DEBUG_ENV, SCHEDULER_DEBUG_ENV_LEGACY))


def _print_scheduler_debug(message: str):
    if not _scheduler_debug_enabled():
        return
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(message, flush=True)


def _set_allgather_allocation_metadata(graph: Graph):
    world_size = None
    for node in graph.nodes:
        if node.target == torch.ops.dc.allgather_param.default:
            if world_size is None:
                world_size = _world_size()
            dtype = node.kwargs.get("dtype") if isinstance(node.kwargs, dict) else None
            profiled_bytes = allgather_allocation_bytes(node.meta.get("tensor_size", 0), dtype, world_size)
            node.meta["allgather_allocation_bytes"] = max(int(node.meta.get("allgather_allocation_bytes", 0) or 0),
                                                          profiled_bytes)


def _scheduler_budget_disabled_reason(graph: Graph, scheduler_budget):
    if scheduler_budget is not None:
        return None
    if not _operator_profile_complete(graph):
        return "incomplete_operator_profile"
    return "invalid_profiled_non_gathered_peak"


def _scheduler_budget_from_operator_profile(gm: GraphModule):
    if not dist.is_initialized():
        return None, "non_distributed"

    _set_allgather_allocation_metadata(gm.graph)
    operator_profile_complete = _sync_profile_complete(_operator_profile_complete(gm.graph))
    if not operator_profile_complete:
        max_gathered_bytes = max_possible_gathered_bytes(gm.graph)
        scheduler_budget = _build_scheduler_budget_from_partial_operator_profile(gm.graph)
        scheduler_budget = _cap_incomplete_profile_budget(gm.graph, scheduler_budget)
        if scheduler_budget is not None and scheduler_budget.max_gathered_bytes < max_gathered_bytes:
            return scheduler_budget, None
        scheduler_budget = SchedulerMemoryBudget.from_available_memory(_rank_min_available_memory(), 0)
        scheduler_budget = _cap_incomplete_profile_budget(gm.graph, scheduler_budget)
        if scheduler_budget is not None and scheduler_budget.max_gathered_bytes >= max_gathered_bytes:
            return None, "incomplete_operator_profile_budget_not_constraining"
        return scheduler_budget, _scheduler_budget_disabled_reason(gm.graph, scheduler_budget)

    scheduler_budget = _build_scheduler_budget_from_operator_profile(gm.graph)
    if scheduler_budget is not None and scheduler_budget.max_gathered_bytes >= max_possible_gathered_bytes(gm.graph):
        return None, "budget_not_constraining"
    return scheduler_budget, _scheduler_budget_disabled_reason(gm.graph, scheduler_budget)


def _log_scheduler_result(graph_id: int, bwd: bool, scheduler_budget, disabled_reason, graph: Graph):
    diagnostics = getattr(graph, SCHEDULER_BUDGET_DIAGNOSTICS_ATTR, {})
    selected = diagnostics.get("selected", [])
    max_live_gathered_bytes = max((entry.get("peak_gathered_bytes", 0) for entry in selected), default=0)
    if scheduler_budget is None:
        _print_scheduler_debug(f"DeepCompile ZeRO-3 scheduler graph_id={graph_id} bwd={bwd} budget_enabled=False "
                               f"disabled_reason={disabled_reason} selected_count={len(selected)} "
                               f"max_live_gathered_bytes={max_live_gathered_bytes}")
        return

    _print_scheduler_debug(
        f"DeepCompile ZeRO-3 scheduler graph_id={graph_id} bwd={bwd} budget_enabled=True "
        f"budget_source={scheduler_budget.source} max_gathered_bytes={scheduler_budget.max_gathered_bytes} "
        f"safety_margin={scheduler_budget.safety_margin} "
        f"profiled_non_gathered_peak_mem={scheduler_budget.profiled_non_gathered_peak_mem} "
        f"budget_rejections={diagnostics.get('budget_rejections', 0)} "
        f"over_budget_fallbacks={len(diagnostics.get('budget_overflows', []))} "
        f"max_live_gathered_bytes={max_live_gathered_bytes}")


def _dtype_element_size(dtype: torch.dtype):
    return torch.empty((), dtype=dtype).element_size()


def _param_allgather_allocation_bytes(param, dtype: torch.dtype):
    return int(param.numel) * _dtype_element_size(dtype)


def add_allgather(graph_id: int,
                  graph: Graph,
                  node: Node,
                  ds_id: int,
                  dtype: torch.dtype,
                  allgather_allocation_bytes: int = None):
    new_ag_node = add_postprocess(graph,
                                  node,
                                  torch.ops.dc.allgather_param.default,
                                  extra_args=[graph_id, ds_id],
                                  extra_kwargs={"dtype": dtype},
                                  name=f"allgather_ds_param_{node.target}_{ds_id}",
                                  meta=_make_node_meta(node, ds_id, True))
    if allgather_allocation_bytes is not None:
        new_ag_node.meta["allgather_allocation_bytes"] = int(allgather_allocation_bytes)
    new_ag_node.meta["val"] = node.meta["val"].to(dtype)

    # Set the previous node back to output
    # We don't want to change the output node to allgather
    output_node = get_output_node(graph)
    output_node.replace_input_with(new_ag_node, node)

    # Add wait as well
    new_wait_node = add_postprocess(graph,
                                    new_ag_node,
                                    torch.ops.dc.wait_allgather.default,
                                    extra_args=[graph_id, ds_id],
                                    name=f"wait_allgather_ds_param__{node.target}_{ds_id}",
                                    meta=_make_node_meta(node, ds_id, False))
    new_wait_node.meta["val"] = new_ag_node.meta["val"]

    return new_ag_node


def add_release(graph_id: int, graph: Graph, node: Node, release_node: Node, ds_id: int, n_users: int):
    new_node = add_postprocess(graph,
                               node,
                               torch.ops.dc.release_param.default,
                               extra_args=[graph_id, ds_id, n_users],
                               name=f"release_ds_param_{release_node.target}_{node.name}_{ds_id}",
                               meta=_make_node_meta(node, ds_id, False))
    new_node.meta["val"] = None


def add_reduce(graph_id: int, graph: Graph, grad_node: Node, param_name: str, ds_id: int):
    new_node = add_postprocess(graph,
                               grad_node,
                               torch.ops.dc.reduce_grad.default,
                               extra_args=[graph_id, ds_id],
                               name=f"reduce_ds_param_{param_name}",
                               meta=_make_node_meta(grad_node, ds_id, True))
    new_node.meta["val"] = None


def add_gather_and_release(graph_id: int, graph: Graph, param_manager, param_nodes: List[Node]) -> Graph:

    node_to_uses = get_real_uses(graph)
    for pn in param_nodes:
        if len(pn.users) == 0:
            continue

        # If the only use of the parameter is a type-cast to a smaller type, fuse it with all-gather.
        fuse_typecast = False
        target_dtype = param_manager.params[pn.name].dtype
        if len([user for user in pn.users if user.op != "output"]) == 1:
            typecast_node = next(iter(pn.users))

            is_cast, casted_dtype = is_cast_op(typecast_node)
            if is_cast and casted_dtype.itemsize < target_dtype.itemsize:
                fuse_typecast = True
                target_dtype = casted_dtype

        param = param_manager.params[pn.name]
        allgather_node = add_allgather(graph_id,
                                       graph,
                                       pn,
                                       param_manager.ds_ids[pn.name],
                                       target_dtype,
                                       allgather_allocation_bytes=_param_allgather_allocation_bytes(
                                           param, target_dtype))
        if fuse_typecast:
            users = node_to_uses[typecast_node]
            wait_node = typecast_node.args[0]
            for user in list(typecast_node.users.keys()):
                if user.op == "output":
                    wait_node.meta["original_output_name"] = typecast_node.name
                user.replace_input_with(typecast_node, wait_node)
            graph.erase_node(typecast_node)
        else:
            users = node_to_uses[pn]
            if len(users) == 0:
                output_node = get_output_node(graph)
                wait_node = next(user for user in allgather_node.users
                                 if user.target == torch.ops.dc.wait_allgather.default)
                wait_node.meta["original_output_name"] = pn.name
                output_node.replace_input_with(pn, wait_node)

        ds_id = param_manager.ds_ids[pn.name]
        for user in users:
            # release_param() only accepts tensors as its first argument. If
            # `user` is a tuple, we should release the param after any of
            # operator.getitem of that tuple.
            #
            # Since no torch op takes a tuple as an input, we simply walk
            # through users of `user` and check if there is any call to
            # operator.getitem.
            for secondary_user in user.users:
                if secondary_user.op == "call_function" and secondary_user.target == _operator.getitem:
                    add_release(graph_id, graph, secondary_user, pn, ds_id, len(users))
                    break
            else:
                add_release(graph_id, graph, user, pn, ds_id, len(users))

    return move_primals_to_head(graph)


def add_gather_and_reduce(graph_id: int, graph: Graph, param_manager, param_nodes_bw: List[Node],
                          param_name_to_grad: Dict[str, Node]) -> Graph:

    add_gather_and_release(graph_id, graph, param_manager, param_nodes_bw)

    for param_name in param_manager.param_names:
        if param_name_to_grad[param_name] is None:
            continue
        add_reduce(graph_id, graph, param_name_to_grad[param_name], param_name, param_manager.ds_ids[param_name])

    return move_primals_to_head(graph)


def add_z3_gather_release_fw(gm: GraphModule,
                             graph_id: int,
                             graph_order: List[Tuple[int, bool]],
                             profiling_results,
                             create_inputs_fn,
                             param_manager,
                             debug_log=False) -> GraphModule:

    nz3 = get_deepcompile_handle()

    real_inputs = create_inputs_fn()
    param_indices = profiling_results[graph_id].param_indices

    gm.graph = add_gather_and_release(graph_id, gm.graph, param_manager[graph_id],
                                      get_param_nodes(gm.graph, param_indices))

    nz3.register_graph_z3(graph_id, [v[1] for v in param_indices])  # Need this before profiling

    profiler = ProfilingInterpreter(gm, debug_log=debug_log)
    profiler.run(*real_inputs)
    del profiler
    gc.collect()
    get_accelerator().empty_cache()
    scheduler_budget, disabled_reason = _scheduler_budget_from_operator_profile(gm)

    rank = dist.get_rank()
    graph_index = get_index_by_graph_id(graph_order, graph_id)
    if rank == 0 and debug_log:
        print(f"Fwd before scheduling graph {graph_index} graph_id={graph_id} {gm.graph}")

    for n in gm.graph.nodes:
        is_ds_param = n.name in param_manager[graph_id].ds_ids
        if "val" in n.meta and is_ds_param:
            # Used for Inductor's validation
            n.meta["val"] = torch.empty([0], dtype=n.meta['val'].dtype, device=n.meta['val'].device)

    gm.graph = fast_free_schedule(
        gm.graph,
        get_accelerator().available_memory(),
        0,  # unused
        debug_log=debug_log,
        scheduler_budget=scheduler_budget)
    _log_scheduler_result(graph_id,
                          bwd=False,
                          scheduler_budget=scheduler_budget,
                          disabled_reason=disabled_reason,
                          graph=gm.graph)

    if rank == 0 and debug_log:
        print(f"Fwd after scheduling graph {graph_index} graph_id={graph_id} {gm.graph}")

    return gm


def add_z3_gather_release_bw(gm: GraphModule,
                             graph_id: int,
                             graph_order: List[Tuple[int, bool]],
                             profiling_results,
                             create_inputs_fn,
                             param_manager,
                             debug_log=False) -> GraphModule:

    param_nodes_bw, param_name_to_grad = param_manager[graph_id].get_bwd_mapping(gm.graph)
    gm.graph = add_gather_and_reduce(graph_id, gm.graph, param_manager[graph_id], param_nodes_bw, param_name_to_grad)

    input_nodes = get_input_nodes(gm.graph)
    real_inputs = create_inputs_fn()
    assert len(input_nodes) == len(real_inputs), f"Expected {len(real_inputs)} inputs, got {len(input_nodes)}"

    real_outputs = ProfilingInterpreter(gm, debug_log=debug_log).run(*real_inputs)

    del real_outputs
    gc.collect()
    get_accelerator().empty_cache()
    scheduler_budget, disabled_reason = _scheduler_budget_from_operator_profile(gm)

    rank = dist.get_rank()
    graph_index = get_index_by_graph_id(graph_order, graph_id)
    if rank == 0 and debug_log:
        print(f"Bwd before scheduling graph {graph_index} graph_id={graph_id} {gm.graph}")

    gm.graph = fast_free_schedule(
        gm.graph,
        get_accelerator().available_memory(),
        0,  # unused
        debug_log=debug_log,
        scheduler_budget=scheduler_budget)
    _log_scheduler_result(graph_id,
                          bwd=True,
                          scheduler_budget=scheduler_budget,
                          disabled_reason=disabled_reason,
                          graph=gm.graph)

    add_end_backward(gm.graph, graph_id, should_release_reduce_buckets(graph_order, graph_id))
    replace_reduce_outputs_with_none(gm.graph)

    return gm


def add_z3_gather_release(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                          create_inputs_fn, mem_budget: float, param_manager, bwd: bool) -> GraphModule:
    if bwd:
        return add_z3_gather_release_bw(gm,
                                        graph_id,
                                        graph_order,
                                        profiling_results,
                                        create_inputs_fn,
                                        param_manager,
                                        debug_log=False)
    return add_z3_gather_release_fw(gm,
                                    graph_id,
                                    graph_order,
                                    profiling_results,
                                    create_inputs_fn,
                                    param_manager,
                                    debug_log=False)
