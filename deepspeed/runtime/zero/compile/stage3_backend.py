# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import gc
from collections import defaultdict
from typing import Dict

import torch
from torch.fx import Graph, GraphModule
from torch.fx.passes.graph_drawer import FxGraphDrawer
from functorch.compile import make_boxed_func
import torch.utils._pytree as pytree
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from .fx import get_output_node, add_gather_and_release, add_gather_and_reduce, register_and_add_wait_allgather
from .graph_param import DSGraphParamManager
from .profilers import ProfilingResult
from .profilers.graph_profile import ProfilingInterpreter, MemoryProfilingInterpreter
from .passes import run_opt_passes
from .list_schedule import simple_prefetch, fast_free_schedule
from .util import get_input_nodes, get_param_nodes, NodeValueOffloadHelper, materialize_fake, count_inflight_values
from .partitioner import get_wrapped_partitioner

graph_counts = defaultdict(int)
param_manager: Dict[int, DSGraphParamManager] = {}
graph_order = []


def reset_graph_order():
    global graph_order
    graph_order = []


def _set_time_and_tensor_size(graph_id, graph: Graph, bwd, profiling_results):
    node_time = []
    tensor_sizes = []

    for n in graph.nodes:
        node_time.append((n.name, n.meta["device_time"] if "device_time" in n.meta else 0.0,
                          n.meta["wall_time"] if "wall_time" in n.meta else 0.0))
        tensor_sizes.append((n.name, n.meta["tensor_size"] if "tensor_size" in n.meta else 0))

    if bwd:
        profiling_results[graph_id].bwd_graph = graph
        profiling_results[graph_id].bwd_time = node_time
        profiling_results[graph_id].bwd_tensor_sizes = tensor_sizes
    else:
        profiling_results[graph_id].fwd_graph = graph
        profiling_results[graph_id].fwd_time = node_time
        profiling_results[graph_id].fwd_tensor_sizes = tensor_sizes


def dump_graph(graph: GraphModule, name: str, skip=False):
    if not skip and dist.get_rank() == 0:
        global graph_counts
        fname = f"{name}_{graph_counts[name]}"
        # graph.graph.print_tabular()

        g = FxGraphDrawer(graph, fname)
        with open(f"{fname}.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())

        graph_counts[name] += 1


profiling_results: Dict[int, ProfilingResult] = {}

enable_opt_passes = False


def launch_opt_passes():
    global enable_opt_passes
    enable_opt_passes = True
    reset_graph_order()


def make_stage3_backend(opt_passes, scheduler, dump_graphs=False, debug_log=False):
    from deepspeed.ops.op_builder import NativeZ3Builder
    nz3 = NativeZ3Builder().load()
    rank = dist.get_rank()

    if scheduler == "simple_prefetch":
        scheduler_fn = simple_prefetch
    elif scheduler == "fast_free":
        scheduler_fn = fast_free_schedule
    else:
        raise ValueError(f"Unknown scheduler {scheduler}")

    def stage3_backend(gm: GraphModule, real_inputs):
        graph_id = id(gm.graph)

        offload_helper = NodeValueOffloadHelper(torch.device(get_accelerator().current_device()))
        needs_backward = pytree.tree_any(lambda x: x.requires_grad if torch.is_tensor(x) else False, real_inputs)

        global graph_order
        graph_order.append((graph_id, needs_backward))

        if len(list(gm.named_parameters())) == 0:
            param_indices = [(i, input_val.ds_id, input_val.ds_shape) for i, input_val in enumerate(real_inputs)
                             if hasattr(input_val, 'ds_id')]
        else:
            # < v2.5
            param_indices = [(i, param.ds_id, param.ds_shape) for i, (n, param) in enumerate(gm.named_parameters())]

        global profiling_results
        if graph_id not in profiling_results:
            profiling_results[graph_id] = ProfilingResult()
            profiling_results[graph_id].param_indices = param_indices
            profiling_results[graph_id].needs_backward = needs_backward

        def fw(gm, sample_inputs):
            if rank == 0 and debug_log:
                print(f"Fwd initial graph graph_id={graph_id} {gm.graph}")

            param_manager[graph_id] = DSGraphParamManager(gm.graph, real_inputs, param_indices)
            original_output_names = [n.name for n in get_output_node(gm.graph).args[0]]

            add_gather_and_release(graph_id, gm.graph, param_manager[graph_id],
                                   get_param_nodes(gm.graph, param_indices))

            nz3.register_graph(graph_id, [v[1] for v in param_indices])  # Need this before profiling
            profiler = ProfilingInterpreter(nz3, gm, debug_log=False)
            real_outputs = profiler.run(*real_inputs)

            total_activation_size = 0
            if needs_backward:
                nonlocal offload_helper
                output_node = get_output_node(gm.graph)
                mod_output_names = [n.name for n in get_output_node(gm.graph).args[0]]
                output_name_map = {n2: n1 for n1, n2 in zip(original_output_names, mod_output_names)}
                for n, v in zip(output_node.args[0], real_outputs):
                    # Save intermediate values on CPU for backward
                    # We don't move ds parameters
                    offload_helper.save(output_name_map[n.name], v, not hasattr(v, 'ds_id'))
                    if torch.is_tensor(v):
                        total_activation_size += v.numel() * v.element_size()
                if rank == 0 and debug_log:
                    print(f"Total activation size graph_id={graph_id} {total_activation_size / 1024 / 1024:.2f} MB")
                    ops_with_mem_str = []
                    for n, v in zip(output_node.args[0], real_outputs):
                        if torch.is_tensor(v):
                            size = v.numel() * v.element_size()
                            ops_with_mem_str.append((
                                size,
                                f" fw output {n.name} {size / total_activation_size * 100:.1f}% {v.shape} {v.dtype} {v.device} {size / 1024 / 1024:.2f} MB"
                            ))
                        else:
                            ops_with_mem_str.append((0, f" fw output {n.name} {v}"))
                    ops_with_mem_str.sort(key=lambda x: x[0], reverse=True)
                    print("\n".join([x[1] for x in ops_with_mem_str]))

            del profiler
            gc.collect()
            get_accelerator().empty_cache()

            if rank == 0 and debug_log:
                print(f"Fwd before scheduling graph graph_id={graph_id} {gm.graph}")

            gm.graph = scheduler_fn(gm.graph,
                                    get_accelerator().available_memory(),
                                    total_activation_size,
                                    debug_log=debug_log)
            gm.recompile()

            if rank == 0 and debug_log:
                count_inflight_values(gm.graph, f"fwd_{graph_id}_inflight_values.csv")

            _, ag_wait_nodes = register_and_add_wait_allgather(graph_id, gm.graph, False)
            nz3.register_graph_ops(graph_id, [n.name for n in ag_wait_nodes], [len(n.args) for n in ag_wait_nodes])

            if rank == 0 and debug_log:
                print(f"Fwd after scheduling graph_id={graph_id} {gm.graph}")

            dump_graph(gm, f"forward_aot_scheduled_{graph_id}", skip=not dump_graphs)

            mem_prof = MemoryProfilingInterpreter(nz3, gm)
            mem_prof.run(*real_inputs)

            if debug_log and rank == 0:
                mem_prof.dump(f"mem_prof_fwd_{graph_id}.csv")
            profiling_results[graph_id].fwd_mem = mem_prof.mem_record
            del mem_prof

            _set_time_and_tensor_size(graph_id, gm.graph, False, profiling_results)

            gc.collect()
            get_accelerator().empty_cache()

            if enable_opt_passes:
                gm = run_opt_passes(nz3, graph_id, gm, real_inputs, opt_passes, graph_order, profiling_results,
                                    param_manager, False, debug_log and rank == 0)

            return make_boxed_func(gm.forward)

        def bw(gm, sample_inputs):
            if rank == 0 and debug_log:
                print(f"Bwd initial graph graph_id={graph_id} {gm.graph}")

            assert graph_id in param_manager, f"Graph {graph_id} not found in param_manager"
            param_nodes_bw, param_name_to_grad = param_manager[graph_id].get_bwd_mapping(gm.graph)

            add_gather_and_reduce(graph_id, gm.graph, param_manager[graph_id], param_nodes_bw, param_name_to_grad)

            input_nodes = get_input_nodes(gm.graph)
            assert len(input_nodes) == len(
                sample_inputs), f"Expected {len(sample_inputs)} inputs, got {len(input_nodes)}"

            nonlocal offload_helper
            validated_inputs = []
            for in_node, in_val in zip(input_nodes, sample_inputs):
                if offload_helper.has_value(in_node.name):
                    validated_inputs.append(offload_helper.get_offloaded_value(in_node.name))
                else:
                    # Here we materialize the fake value on CPU to reduce the peak memory
                    # The values are moved to the device memory in the profiler
                    validated_inputs.append(materialize_fake(in_val, device="cpu"))
            validated_inputs = tuple(validated_inputs)

            real_outputs = ProfilingInterpreter(nz3, gm, debug_log=False).run(*validated_inputs)

            output_size = sum(v.numel() * v.element_size() for v in real_outputs if torch.is_tensor(v))
            if rank == 0 and debug_log:
                print(f"Total backward grad size graph_id={graph_id} {output_size / 1024 / 1024:.2f} MB")
                ops_with_mem_str = []
                output_node = get_output_node(gm.graph)
                for n, v in zip(output_node.args[0], real_outputs):
                    if torch.is_tensor(v):
                        size = v.numel() * v.element_size()
                        ops_with_mem_str.append((
                            size,
                            f" bw output {n.name} {size / output_size * 100:.1f}% {v.shape} {v.dtype} {v.device} {size / 1024 / 1024:.2f} MB"
                        ))
                    elif v is not None:
                        ops_with_mem_str.append((0, f" bw output {n.name} {v}"))
                ops_with_mem_str.sort(key=lambda x: x[0], reverse=True)
                print("\n".join([x[1] for x in ops_with_mem_str]))

            del real_outputs
            offload_helper.clear()
            gc.collect()
            get_accelerator().empty_cache()

            if rank == 0 and debug_log:
                print(f"Bwd before scheduling graph graph_id={graph_id} {gm.graph}")

            gm.graph = scheduler_fn(gm.graph, get_accelerator().available_memory(), output_size, debug_log=debug_log)

            if rank == 0 and debug_log:
                print(f"Bwd after scheduling graph_id={graph_id} {gm.graph}")

            if rank == 0 and debug_log:
                count_inflight_values(gm.graph, f"bwd_{graph_id}_inflight_values.csv")

            _, ag_wait_nodes = register_and_add_wait_allgather(graph_id, gm.graph, True)
            nz3.register_bwd_graph_ops(graph_id, [n.name for n in ag_wait_nodes], [len(n.args) for n in ag_wait_nodes])
            dump_graph(gm, f"backward_aot_scheduled_{graph_id}", skip=not dump_graphs)
            gm.recompile()

            mem_prof = MemoryProfilingInterpreter(nz3, gm)
            mem_prof.run(*validated_inputs)
            if debug_log and rank == 0:
                mem_prof.dump(f"mem_prof_bwd_{graph_id}.csv")
            profiling_results[graph_id].bwd_mem = mem_prof.mem_record

            del mem_prof
            gc.collect()
            get_accelerator().empty_cache()

            _set_time_and_tensor_size(graph_id, gm.graph, True, profiling_results)

            global enable_opt_passes
            if enable_opt_passes:
                gm = run_opt_passes(nz3, graph_id, gm, validated_inputs, opt_passes, graph_order, profiling_results,
                                    param_manager, True, debug_log and rank == 0)

            return make_boxed_func(gm.forward)

        # Call AOTAutograd
        aot_mod = aot_module_simplified(gm,
                                        real_inputs,
                                        fw_compiler=fw,
                                        bw_compiler=bw,
                                        partition_fn=get_wrapped_partitioner(param_indices))
        aot_mod = torch._dynamo.optimize()(aot_mod)

        return aot_mod

    return stage3_backend
