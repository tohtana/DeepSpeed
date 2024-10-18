# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import gc
from collections import defaultdict
from typing import List, Dict

import torch
from torch.fx import Node, Graph, GraphModule
from torch.fx.passes.graph_drawer import FxGraphDrawer
from functorch.compile import make_boxed_func
import torch.utils._pytree as pytree
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.compile.tracer import add_dependency_on_params
from deepspeed.runtime.zero.compile.nx import fx_to_nx, find_reachable_terminal_nodes

from .fx import add_postprocess, add_args_process, get_output_node
# from .schedule import schedule
from .graph_param import DSGraphParamManager
from .profile import ProfilingInterpreter
from .list_schedule import list_schedule2
from .util import get_input_nodes, get_param_nodes, NodeValueOffloadHelper, materialize_fake, count_inflight_values
from .tracer import ops_no_release, ops_no_wait
from .partitioner import get_wrapped_partitioner

import os

pid = os.getpid()

gathered_params = {}
param_map = {}
z3_optimizer = None
nz3 = None


def _make_node_meta(node: Node, ds_id: int, comm: bool):
    meta = {"param_name": node.name, "ds_id": ds_id, "comm": comm}
    if "tensor_meta" in node.meta:
        meta["tensor_meta"] = node.meta["tensor_meta"]
    return meta


def add_allgather(graph_id: int, graph: Graph, node: Node, ds_id: int):
    new_node = add_postprocess(graph,
                               node,
                               torch.ops.native_z3.allgather_param,
                               extra_args=[graph_id, ds_id],
                               name=f"allgather_ds_param_{node.target}_{ds_id}",
                               meta=_make_node_meta(node, ds_id, True))
    output_node = get_output_node(graph)
    output_node.replace_input_with(new_node, node)


def add_release(graph_id: int, graph: Graph, node: Node, release_node: Node, ds_id: int, count: int):
    add_postprocess(graph,
                    node,
                    torch.ops.native_z3.release_param,
                    extra_args=[graph_id, ds_id, count],
                    name=f"release_ds_param_{release_node.target}_{node.name}_{ds_id}",
                    meta=_make_node_meta(node, ds_id, False))


def add_wait_allgather(graph_id: int, graph: Graph, node: Node, ds_id: int, user: str, n_args: int, bwd: bool):
    add_args_process(graph,
                     node,
                     torch.ops.native_z3.wait_allgather,
                     extra_args=[graph_id, ds_id, user, n_args, bwd],
                     name=f"wait_allgather_ds_param_{ds_id}",
                     meta=_make_node_meta(node, ds_id, False))


def add_reduce(graph_id: int, graph: Graph, grad_node: Node, param_name: str, ds_id: int):
    add_postprocess(graph,
                    grad_node,
                    torch.ops.native_z3.reduce_grad,
                    extra_args=[graph_id, ds_id],
                    name=f"reduce_ds_param_{param_name}",
                    meta=_make_node_meta(grad_node, ds_id, True))


def register_and_add_wait_allgather(graph_id: int, graph: Graph, bwd: bool):

    ds_ids = []
    ag_wait_nodes = []

    for node in graph.nodes:
        ag_args = [
            arg for arg in node.args if isinstance(arg, Node) and arg.target == torch.ops.native_z3.allgather_param
        ]
        if len(ag_args) > 0:
            if node.target in ops_no_wait:
                continue

            assert len(ag_args) == 1, f"Node {node.name} takes multiple allgathered params"
            ag_wait_nodes.append(node)

            ds_id = ag_args[0].meta["ds_id"]
            add_wait_allgather(graph_id, graph, node, ds_id, node.name, len(node.args), bwd)
            ds_ids.append(ds_id)

    return ds_ids, ag_wait_nodes


def add_gather_and_release(graph_id: int, graph: Graph, param_manager: DSGraphParamManager, param_nodes: List[Node]):
    add_dependency_on_params(graph, param_nodes)

    nx_graph = fx_to_nx(graph)
    last_user_nodes = {}
    for pn in param_nodes:
        dependent_nodes = [n for n in graph.nodes if pn in n.required_inputs]
        last_user_nodes[pn] = find_reachable_terminal_nodes(nx_graph, dependent_nodes)

    for pn in param_nodes:
        add_allgather(graph_id, graph, pn, param_manager.ds_ids[pn.name])

    release_counts = defaultdict(int)
    to_add = []
    for pn, nodes in last_user_nodes.items():
        for node in nodes:
            if node.target not in ops_no_release:
                to_add.append((node, pn, param_manager.ds_ids[pn.name]))
                release_counts[pn.name] += 1

    for node, pn, ds_id in to_add:
        add_release(graph_id, graph, node, pn, ds_id, release_counts[pn.name])


def add_gather_and_reduce(graph_id: int, graph: Graph, param_manager: DSGraphParamManager, param_nodes_bw: List[Node],
                          param_name_to_grad: Dict[str, Node]):

    add_gather_and_release(graph_id, graph, param_manager, param_nodes_bw)

    for param_name in param_manager.param_names:
        add_reduce(graph_id, graph, param_name_to_grad[param_name], param_name, param_manager.ds_ids[param_name])


graph_counts = defaultdict(int)
param_manager = {}


def dump_graph(graph: GraphModule, name: str, skip=False):
    if not skip and dist.get_rank() == 0:
        global graph_counts
        fname = f"{name}_{graph_counts[name]}"
        graph.graph.print_tabular()

        g = FxGraphDrawer(graph, fname)
        with open(f"{fname}.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())

        graph_counts[name] += 1


def make_stage3_backend(dump_graphs=False, debug_log=True):
    from deepspeed.ops.op_builder import NativeZ3Builder
    nz3 = NativeZ3Builder().load()
    rank = dist.get_rank()

    def stage3_backend(gm: GraphModule, real_inputs):
        graph_id = id(gm.graph)

        offload_helper = NodeValueOffloadHelper(torch.device(get_accelerator().current_device()))
        needs_backward = pytree.tree_any(lambda x: x.requires_grad if torch.is_tensor(x) else False, real_inputs)

        if len(list(gm.named_parameters())) == 0:
            param_indices = [(i, input_val.ds_id, input_val.ds_shape) for i, input_val in enumerate(real_inputs)
                             if hasattr(input_val, 'ds_id')]
        else:
            # < v2.5
            param_indices = [(i, param.ds_id, param.ds_shape) for i, (n, param) in enumerate(gm.named_parameters())]

        def fw(gm, sample_inputs):

            if rank == 0 and dump_graph:
                print(f"Initial graph graph_id={graph_id} {gm.graph}")

            param_manager[graph_id] = DSGraphParamManager(gm.graph, sample_inputs, param_indices)
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

            if rank == 0 and dump_graph:
                print(f"Before scheduling graph graph_id={graph_id} {gm.graph}")

            gm.graph = list_schedule2(gm.graph,
                                      get_accelerator().available_memory(),
                                      total_activation_size,
                                      debug_log=debug_log)

            if rank == 0 and debug_log:
                count_inflight_values(gm.graph, f"fwd_{graph_id}_inflight_values.csv")

            _, ag_wait_nodes = register_and_add_wait_allgather(graph_id, gm.graph, False)
            nz3.register_graph_ops(graph_id, [n.name for n in ag_wait_nodes], [len(n.args) for n in ag_wait_nodes])

            if rank == 0 and dump_graph:
                print(f"After scheduling graph_id={graph_id} {gm.graph}")

            dump_graph(gm, f"forward_aot_scheduled", skip=not dump_graphs)

            gc.collect()
            get_accelerator().empty_cache()

            gm.recompile()
            return make_boxed_func(gm.forward)

        def bw(gm, sample_inputs):
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

            offload_helper.clear()
            gc.collect()
            get_accelerator().empty_cache()

            gm.graph = list_schedule2(gm.graph, get_accelerator().available_memory(), output_size, debug_log=debug_log)

            if rank == 0 and debug_log:
                count_inflight_values(gm.graph, f"bwd_{graph_id}_inflight_values.csv")

            _, ag_wait_nodes = register_and_add_wait_allgather(graph_id, gm.graph, True)
            nz3.register_bwd_graph_ops(graph_id, [n.name for n in ag_wait_nodes], [len(n.args) for n in ag_wait_nodes])
            dump_graph(gm, f"backward_aot_scheduled", skip=not dump_graphs)
            gm.recompile()
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
