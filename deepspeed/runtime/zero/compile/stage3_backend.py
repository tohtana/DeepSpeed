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
from deepspeed.runtime.utils import see_memory_usage

from .fx import add_postprocess, add_args_process, get_output_node
# from .schedule import schedule
from .graph_param import DSGraphParamManager
from .profile import ProfilingInterpreter
from .list_schedule import list_schedule2
from .util import get_input_nodes, get_param_nodes, NodeValueOffloadHelper, materialize_fake
from .tracer import ops_no_release, ops_no_wait
from .partitioner import get_wrapped_partitioner

import os
import time

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


def make_stage3_backend(dump_graphs=False):
    from deepspeed.ops.op_builder import NativeZ3Builder
    nz3 = NativeZ3Builder().load()
    rank = dist.get_rank()
    print(f"Rank {rank} pid {pid} make_stage3_backend")

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

            time_start = time.time()
            if rank == 0:
                see_memory_usage(f"Compiling fw starting", force=True)

            param_manager[graph_id] = DSGraphParamManager(gm.graph, sample_inputs, param_indices)
            original_output_names = [n.name for n in get_output_node(gm.graph).args[0]]

            add_gather_and_release(graph_id, gm.graph, param_manager[graph_id],
                                   get_param_nodes(gm.graph, param_indices))

            time_add_gather_release = time.time()
            if rank == 0:
                see_memory_usage(f"Compiling fw add_gather_and_release time={time_add_gather_release - time_start}", force=True)

            nz3.register_graph(graph_id, [v[1] for v in param_indices])  # Need this before profiling
            profiler = ProfilingInterpreter(nz3, gm)
            real_outputs = profiler.run(*real_inputs)

            time_profiling = time.time()
            if rank == 0:
                see_memory_usage(f"Compiling fw profiling time={time_profiling - time_add_gather_release}", force=True)

            if needs_backward:
                nonlocal offload_helper
                output_node = get_output_node(gm.graph)
                mod_output_names = [n.name for n in get_output_node(gm.graph).args[0]]
                output_name_map = {n2: n1 for n1, n2 in zip(original_output_names, mod_output_names)}
                # non_param_input_names = [n.name for n in get_input_nodes(gm.graph)][:len(param_indices)]
                for n, v in zip(output_node.args[0], real_outputs):
                    # Save intermediate values on CPU for backward
                    # if rank == 0:
                    #     print(f" n.name={n.name} offload?={n.name not in non_param_input_names} non_param_input_names={non_param_input_names}")
                    # offload_helper.save(output_name_map[n.name], v, n.name not in non_param_input_names)
                    offload_helper.offload(output_name_map[n.name], v)
            time_offload = time.time()
            if rank == 0:
                see_memory_usage(f"Compiling fw offload time={time_offload - time_profiling}", force=True)

            gm.graph = list_schedule2(gm.graph)

            time_list_schedule = time.time()
            if rank == 0:
                see_memory_usage(f"Compiling fw list_schedule time={time_list_schedule - time_offload}", force=True)

            _, ag_wait_nodes = register_and_add_wait_allgather(graph_id, gm.graph, False)

            time_register_and_add_wait_allgather = time.time()
            if rank == 0:
                see_memory_usage(
                    f"Compiling fw register_and_add_wait_allgather time={time_register_and_add_wait_allgather - time_list_schedule}", force=True
                )

            nz3.register_graph_ops(graph_id, [n.name for n in ag_wait_nodes], [len(n.args) for n in ag_wait_nodes])
            dump_graph(gm, f"forward_aot_scheduled", skip=not dump_graphs)
        
            gc.collect()
            get_accelerator().empty_cache()

            if rank == 0:
                see_memory_usage(f"Compiling fw end", force=True)
                for o, n in zip(real_outputs, get_output_node(gm.graph).args[0]):
                    if torch.is_tensor(o):
                        print(f"Output tensor {n.name} {o.size()} {o.device}")
                    else:
                        print(f"Output {n.name} {o}")

                # print(f"Graph {graph_id} compiled {gm.graph}")

            gm.recompile()
            return make_boxed_func(gm.forward)

        def bw(gm, sample_inputs):
            time_start = time.time()

            assert graph_id in param_manager, f"Graph {graph_id} not found in param_manager"
            param_nodes_bw, param_name_to_grad = param_manager[graph_id].get_bwd_mapping(gm.graph)

            add_gather_and_reduce(graph_id, gm.graph, param_manager[graph_id], param_nodes_bw, param_name_to_grad)

            time_add_gather_and_reduce = time.time()
            if rank == 0:
                see_memory_usage(f"Compiling bw add_gather_and_reduce time={time_add_gather_and_reduce - time_start}", force=True)

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

            time_materialize_fake = time.time()
            if rank == 0:
                see_memory_usage(f"Compiling bw materialize_fake time={time_materialize_fake - time_add_gather_and_reduce}", force=True)

            ProfilingInterpreter(nz3, gm).run(*validated_inputs)

            time_profiling = time.time()
            if rank == 0:
                see_memory_usage(f"Compiling bw profiling time={time_profiling - time_materialize_fake}", force=True)

            offload_helper.clear()
            gc.collect()
            get_accelerator().empty_cache()

            gm.graph = list_schedule2(gm.graph)

            time_list_schedule = time.time()
            if rank == 0:
                see_memory_usage(f"Compiling bw list_schedule time={time_list_schedule - time_profiling}", force=True)

            _, ag_wait_nodes = register_and_add_wait_allgather(graph_id, gm.graph, True)

            time_register_and_add_wait_allgather = time.time()
            if rank == 0:
                see_memory_usage(
                    f"Compiling bw register_and_add_wait_allgather time={time_register_and_add_wait_allgather - time_list_schedule}", force=True
                )

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
