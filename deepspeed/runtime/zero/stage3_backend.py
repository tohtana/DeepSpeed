from typing import Dict
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch._dynamo
from torch.fx.passes.graph_drawer import FxGraphDrawer
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified

from deepspeed.runtime.zero.compile.tracer import add_dependency_on_params
from deepspeed.runtime.zero.compile.nx import fx_to_nx, find_reachable_terminal_nodes, sort_nodes_by_distance_to_output, serialize
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from typing import Callable, Any, List, Set
import torch
from torch.fx import Node, Graph, GraphModule

from .compile.fx import add_postprocess
from .compile.schedule import schedule
from .compile.graph_param import DSGraphParamManager

import os
pid = os.getpid()

gathered_params = {}
param_map = {}
z3_optimizer = None


def make_allgather_func(name: str):

    # torch.library.define(f"ds_compile::gather_param_{name}", "(Tensor x) -> Tensor")

    def allgather_param(x):
        # print(f"allgather_param {name} {x.__class__} ds_id={hasattr(x, 'ds_id')} ds_param={hasattr(x, 'ds_param')} all_gather={hasattr(x, 'all_gather')}")
        if hasattr(x, 'ds_id'):
            x.all_gather(param_list=[x])
            global gathered_params
            gathered_params[name] = x

            global param_map
            param_map[name] = x
        return x

    # torch.library.impl(f"ds_compile::gather_param_{name}", ["cpu", "cuda"], allgather_param)
    return allgather_param


def make_release_func(name: str):

    def release_param(*x):
        global gathered_params
        if name in gathered_params:
            param = gathered_params.pop(name)
            param.partition(param_list=[param], has_been_updated=False)
        if len(x) == 1:
            x = x[0]
        return x

    return release_param


def make_reduce_func(name: str):

    def reduce_grad(*x):
        if name in param_map:
            param = param_map[name]
            if param.ds_status != ZeroParamStatus.AVAILABLE:
                 param.all_gather(param_list=[param])
            param.grad = x[0]
            z3_optimizer.reduce_ready_partitions_and_remove_grads(param)
            param.partition(param_list=[param], has_been_updated=False)
            return None
        if len(x) == 1:
            x = x[0]
        return x

    return reduce_grad


def add_allgather(graph: Graph, node: Node):
    return add_postprocess(graph, node, make_allgather_func(node.target))


def add_release(graph: Graph, node: Node, release_node: Node):
    return add_postprocess(graph, node, make_release_func(release_node.target))


def add_reduce(graph: Graph, grad_node: Node, param_name: str):
    return add_postprocess(graph, grad_node, make_reduce_func(param_name))

def add_gather_and_release(gm: GraphModule, param_nodes: List[Node]):
    graph = gm.graph
    add_dependency_on_params(graph, param_nodes)

    nx_graph = fx_to_nx(graph)
    user_nodes = {}
    for pn in param_nodes:
        dependent_nodes = [n for n in graph.nodes if pn in n.required_inputs]
        user_nodes[pn] = find_reachable_terminal_nodes(nx_graph, dependent_nodes)
        
    allgather_nodes = {}
    for pn in param_nodes:
        allgather_nodes[pn] = add_allgather(graph, pn)

    release_nodes = {}
    for v, nodes in user_nodes.items():
        for node in nodes:
            assert v not in release_nodes
            release_nodes[v] = add_release(graph, node, v)

    return allgather_nodes, release_nodes    


def add_gather_and_reduce(gm: GraphModule, param_manager: DSGraphParamManager):
    for pn in param_manager.param_nodes_bw:
        n = add_allgather(gm.graph, pn)
        param_manager.add_allgather_node(pn.name, n, bw=True)
    for pn in param_manager.param_nodes:
        rn = add_reduce(gm.graph, param_manager.get_grad_name(pn.name), pn.name)
        param_manager.add_release_node(pn.name, rn, bw=True)
        

graph_counts = defaultdict(int)
param_manager = None


def dump_graph(graph: GraphModule, name: str, skip=False):
    if not skip:
        global graph_counts
        fname = f"{name}_{graph_counts[name]}.dot"

        g = FxGraphDrawer(graph, fname)
        with open(f"{name}.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())

        graph_counts[name] += 1


def make_stage3_backend(dump_graphs=False):

    def stage3_backend(gm: GraphModule, sample_inputs): 
        n_params = len(list(gm.named_parameters()))

        def fw(gm, sample_inputs):
            global param_manager
            param_manager = DSGraphParamManager(gm.graph, sample_inputs, n_params)

            dump_graph(gm, f"forward_aot", skip=not dump_graphs)

            allgather_nodes, release_nodes = add_gather_and_release(gm, param_manager.param_nodes)
            for pn, an in allgather_nodes.items():
                param_manager.add_allgather_node(pn.name, an)
            for pn, rn in release_nodes.items():
                param_manager.add_release_node(pn.name, rn)

            gm.graph = schedule(gm.graph, param_manager)
            dump_graph(gm, f"forward_aot_scheduled", skip=not dump_graphs)

            gm.recompile()
            return make_boxed_func(gm.forward)

        def bw(gm, sample_inputs):
            param_manager.add_bw_graph(gm.graph)

            dump_graph(gm, f"backward_aot", skip=not dump_graphs)

            add_gather_and_reduce(gm, param_manager)

            gm.graph = schedule(gm.graph, param_manager, bw=True)
            dump_graph(gm, f"backward_aot_scheduled", skip=not dump_graphs)

            gm.recompile()
            return make_boxed_func(gm.forward)

        # Call AOTAutograd
        aot_mod = aot_module_simplified(gm, sample_inputs,
                                        fw_compiler=fw,
                                        bw_compiler=bw)
        aot_mod = torch._dynamo.optimize()(aot_mod)

        return aot_mod

    return stage3_backend
