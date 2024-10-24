# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools
import operator
from typing import List, Tuple, Dict
from weakref import WeakSet

import torch
from torch.fx import Node, Graph
from torch.fx.node import map_aggregate, Argument, map_arg

try:
    from torch._subclasses.fake_tensor import unset_fake_temporarily
except ImportError:
    # torch < v2.5
    from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode as unset_fake_temporarily

no_copy_ops = {torch.ops.aten.t.default, torch.ops.aten.view.default}


def get_input_nodes(graph: Graph) -> List[Node]:
    return [n for n in graph.nodes if n.op == "placeholder"]


def get_param_nodes(graph: Graph, index_to_ds_ids: List[Tuple[int, int]]) -> List[Node]:
    all_input_nodes = get_input_nodes(graph)
    return [all_input_nodes[i] for i, _, _ in index_to_ds_ids]


def is_comm_op(node: Node) -> bool:
    return "comm" in node.meta and node.meta["comm"]


def dtype_to_elem_size(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        elem_size = 4
    elif dtype == torch.float64:
        elem_size = 8
    elif dtype == torch.float16:
        elem_size = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return elem_size


def tensor_meta_size(tensor_meta) -> int:
    numel = 1 if len(tensor_meta.shape) == 0 else functools.reduce(operator.mul, tensor_meta.shape)

    dtype = tensor_meta.dtype
    if dtype == torch.float32:
        elem_size = 4
    elif dtype == torch.float64 or dtype == torch.int64:
        elem_size = 8
    elif dtype == torch.float16 or dtype == torch.bfloat16:
        elem_size = 2
    elif dtype == torch.bool:
        elem_size = 1
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return numel * elem_size


class NodeValueOffloadHelper:

    def __init__(self, device):
        self.device = device
        self.env_values: Dict[str, Argument] = {}
        self.env_values_device_unchanged: WeakSet[torch.Tensor] = WeakSet()

    def _to_cpu(self, v):
        if torch.is_tensor(v):
            if v.device == self.device:
                with unset_fake_temporarily():
                    return v.to('cpu').detach()
            else:
                self.env_values_device_unchanged.add(v)
        return v

    def _from_cpu(self, v):
        if torch.is_tensor(v) and v not in self.env_values_device_unchanged:
            return v.to(self.device).detach()
        return v

    def save(self, name: str, v: Argument, offload) -> None:
        self.env_values[name] = map_aggregate(v, lambda x: self._to_cpu(x) if offload else x)

    def load(self, name: str) -> Argument:
        return map_aggregate(self.env_values[name], lambda x: self._from_cpu(x))

    def get_offloaded_value(self, name: str) -> Argument:
        return self.env_values[name]

    def has_value(self, name: str) -> bool:
        return name in self.env_values

    def clear(self) -> None:
        self.env_values.clear()
        self.env_values_device_unchanged.clear()


def materialize_fake(v, device=None):
    from torch._subclasses.fake_tensor import is_fake

    def convert(t):
        if is_fake(t):
            with unset_fake_temporarily():
                if t.is_floating_point():
                    return torch.randn(t.shape,
                                       dtype=t.dtype,
                                       device=t.device if device is None else device,
                                       layout=t.layout,
                                       requires_grad=t.requires_grad,
                                       pin_memory=t.is_pinned())
                else:
                    return torch.zeros(t.shape,
                                       dtype=t.dtype,
                                       device=t.device if device is None else device,
                                       requires_grad=t.requires_grad)

        return t

    return map_aggregate(v, lambda x: convert(x))


def get_last_uses(graph: Graph):
    position = {node: i for i, node in enumerate(graph.nodes)}

    node_to_last_use: Dict[Node, Node] = {}
    user_to_last_uses: Dict[Node, List[Node]] = {}

    def register_last_uses(n: Node, user: Node):
        update = False
        known_last_use = None

        if user.target in no_copy_ops and n in node_to_last_use:
            last_user = node_to_last_use[user]
            last_use_position = position[last_user]

            known_last_use = node_to_last_use[n]
            known_last_use_position = position[known_last_use]
            update = last_use_position > known_last_use_position

        if n not in node_to_last_use or update:
            if user.target in no_copy_ops:
                user = node_to_last_use[user]

            node_to_last_use[n] = user
            user_to_last_uses.setdefault(user, []).append(n)

            if known_last_use:
                user_to_last_uses[known_last_use].remove(n)

    for node in reversed(graph.nodes):
        map_arg(node.args, lambda n: register_last_uses(n, node))
        map_arg(node.kwargs, lambda n: register_last_uses(n, node))

    return node_to_last_use, user_to_last_uses


def count_inflight_values(graph: Graph, file_path: str):
    position = {node: i for i, node in enumerate(graph.nodes)}

    node_to_last_use, user_to_last_uses = get_last_uses(graph)

    max_inflight_size = 0
    inflight_values = set()

    # Output csv.
    csv_filename = file_path
    csv_data = []
    header = [
        'Node', 'tensor_size', 'inflight_size', 'inflight_size_in_output', 'args', 'users', 'node_to_last_use',
        'lifetime', 'user_to_last_uses', 'inflight_values'
    ]
    csv_data.append(header)

    from .fx import get_output_node
    output_node = get_output_node(graph)
    values_in_output = set([n for n in output_node.args[0] if isinstance(n, Node)])

    for node in graph.nodes:
        inflight_values.add(node)
        if node in user_to_last_uses:
            for to_delete in user_to_last_uses[node]:
                inflight_values.remove(to_delete)

        assert "tensor_size" in node.meta, f"Node {node} does not have tensor_size"
        inflight_size = sum(n.meta["tensor_size"] for n in inflight_values)
        inflight_size_in_output = sum(n.meta["tensor_size"] for n in inflight_values if n in values_in_output)

        lifetime = position[node_to_last_use[node]] - position[node] if node in node_to_last_use else 0

        row = [
            node.name, node.meta["tensor_size"], inflight_size, inflight_size_in_output,
            [a.name for a in node.args if isinstance(a, Node)],
            list(node.users.keys()), node_to_last_use[node] if node in node_to_last_use else 'NA', lifetime,
            user_to_last_uses[node] if node in user_to_last_uses else 'NA',
            list(inflight_values)
        ]
        csv_data.append(row)

        # print(
        #     f"Node: {node.name} users: {list(node.users.keys())} node_to_last_use: {node_to_last_use[node] if node in node_to_last_use else 'NA'} user_to_last_uses: {user_to_last_uses[node] if node in user_to_last_uses else 'NA'} inflight_values: {inflight_values} inflight_size: {inflight_size}"
        # )
        max_inflight_size = max(max_inflight_size, inflight_size)

    import csv
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"Max inflight size: {max_inflight_size}")
    print(f"Data successfully written to {csv_filename}")
