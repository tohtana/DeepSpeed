from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from collections import defaultdict
import torch
from torch.fx import Tracer, Node, Graph
from torch.fx.proxy import Proxy, GraphAppendingTracer
import torch.utils._pytree as pytree


aten = torch._ops.ops.aten
ops_reuse_inputs = [
    aten.t.default
]

def add_dependency_on_params(graph: Graph, param_nodes: List[Node]) -> None:
    reuse_inputs = defaultdict(list)

    for node in graph.nodes:
        if node.op == 'call_function':
            if node.target in ops_reuse_inputs:
                for a in node.args:
                    for param, users in reuse_inputs.items():
                        if a in users:
                            users.append(node)
        else:
            if node.op == 'placeholder' and node in param_nodes:
                reuse_inputs[node].append(node)

        node.required_inputs = []
        for a in node.args:
            for param, users in reuse_inputs.items():
                if a in users:
                    node.required_inputs.append(param)


def fake_to_real(a):
    def to_real_tensor(t):
        return t

    return pytree.tree_map_only(torch.Tensor, to_real_tensor, a)



class ParamLifetimeCheckTracer(Tracer):
        
    # Inside here you can override various methods
    # to customize tracing. See the `Tracer` API
    # reference
    def __init__(self):
        super().__init__()
        self.reuse_inputs = defaultdict(list)

    def create_node(self, kind : str, target : Union[str, Callable],
                    args : Tuple[Any], kwargs : Dict[str, Any], name : Optional[str] = None,
                    type_expr : Optional[Any] = None) -> Node:
        n = super().create_node(kind, target, args, kwargs, name)

        if n.target in self.ops_reuse_inputs:
            for a in args:
                if isinstance(a, Node):
                    self.reuse_inputs[a].append(n)
        # n.reuse_inputs = 
        # print(f"create_node kind={kind} target={target} args={args} kwargs={kwargs} name={name} type_expr={type_expr}")
        return n

    def check_lifetime(self, node):
        if node in self.reuse_inputs:
            for user in self.reuse_inputs[node]:
                self.check_lifetime(user)
        print(f"check_lifetime node={node}")


# Let's use this custom tracer to trace through this module
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 3)

    def forward(self, x):
        return torch.relu(x) + torch.ones(3, 4)

# mod = MyModule()

# traced_graph = ParamLifetimeCheckTracer().trace(mod)
# # trace() returns a Graph. Let's wrap it up in a
# # GraphModule to make it runnable
# traced = torch.fx.GraphModule(mod, traced_graph)
