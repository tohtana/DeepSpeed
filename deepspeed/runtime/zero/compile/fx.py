from typing import Callable, Any, List, Set
from torch.fx import Node, Graph, GraphModule


def get_output_node(graph: Graph):
    for v in graph.nodes:
        if v.target == "output":
            return v
    raise ValueError("No output node found")


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
    return new_node