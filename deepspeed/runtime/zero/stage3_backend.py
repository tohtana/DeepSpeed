from copy import deepcopy
import argparse

import torch
import torch._dynamo
from torch.fx.passes.graph_drawer import FxGraphDrawer
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified

from deepspeed.runtime.zero.compile.tracer import add_dependency_on_params
from deepspeed.runtime.zero.compile.nx import fx_to_nx, find_reachable_terminal_nodes

from typing import Callable, Any
import torch
from torch.fx import Node, Graph, GraphModule


import os
pid = os.getpid()

gathered_params = {}

def make_allgather_func(name: str):

    # torch.library.define(f"ds_compile::gather_param_{name}", "(Tensor x) -> Tensor")

    def allgather_param(x):
        if hasattr(x, 'ds_param'):
            x = x.ds_param
            x.all_gather(param_list=[x])
            global gathered_params
            gathered_params[name] = x
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


def get_param_nodes(graph: Graph, n_params: int):
    return [n for n in graph.nodes if n.op == "placeholder"][:n_params]


def get_param_users(graph: Graph, n_params: int):
    param_nodes = get_param_nodes(graph, n_params)
    return {p: set(p.users.keys()) for p in param_nodes}


def add_postprocess(graph: Graph, node: Node, fn: Callable[..., Any]):
    # https://github.com/pytorch/examples/blob/main/fx/wrap_output_dynamically.py
    with graph.inserting_after(node):
        node_users = node.users.keys()
        new_node = graph.call_function(fn, (node,), {})
        users = {}
        for u in node_users:
            if u != new_node:
                users[u] = (node, new_node)
        for u, (old_in, new_in) in users.items():
            u.replace_input_with(old_in, new_in)


def add_allgather(graph: Graph, node: Node):
    add_postprocess(graph, node, make_allgather_func(node.target))


def add_release(graph: Graph, node: Node, release_node: Node):
    add_postprocess(graph, node, make_release_func(release_node.target))


def add_gather_and_release(gm: GraphModule, n_params: int):
    graph = gm.graph
    param_nodes = get_param_nodes(graph, n_params)
    add_dependency_on_params(graph, param_nodes)

    nx_graph = fx_to_nx(graph)
    release_nodes = {}
    for pn in param_nodes:
        dependent_nodes = [n for n in graph.nodes if pn in n.required_inputs]
        release_nodes[pn] = find_reachable_terminal_nodes(nx_graph, dependent_nodes)
        
    for pn in param_nodes:
        add_allgather(graph, pn)

    for v, nodes in release_nodes.items():
        for node in nodes:
            add_release(graph, node, v)

    gm.recompile()


backend_count = 0
fw_count = 0
def stage3_backend(gm: GraphModule, sample_inputs): 
    global backend_count

    n_params = len(list(gm.named_parameters()))

    def fw(gm, sample_inputs):
        # global fw_count
        # # g = FxGraphDrawer(gm, 'fn')
        # # with open(f"forward_aot_{backend_count}_{fw_count}.svg", "wb") as file:
        # #     file.write(g.get_dot_graph().create_svg())

        add_gather_and_release(gm, n_params)

        # # g = FxGraphDrawer(gm, 'fn')
        # # with open(f"forward_aot_{backend_count}_{fw_count}_mod.svg", "wb") as file:
        # #     file.write(g.get_dot_graph().create_svg())

        # fw_count += 1

        return make_boxed_func(gm.forward)

    def bw(gm, sample_inputs):
        add_gather_and_release(gm, n_params)
        return make_boxed_func(gm.forward)

    # Call AOTAutograd
    aot_mod = aot_module_simplified(gm, sample_inputs,
                                    fw_compiler=fw,
                                    bw_compiler=bw)
    aot_mod = torch._dynamo.optimize()(aot_mod)

    backend_count += 1
    return aot_mod

