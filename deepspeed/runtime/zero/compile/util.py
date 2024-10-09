# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools
import operator
from typing import List, Tuple, Dict
from weakref import WeakSet

import torch
from torch.fx import Node, Graph
from torch.fx.node import map_aggregate, Argument

try:
    from torch._subclasses.fake_tensor import unset_fake_temporarily
except ImportError:
    # torch < v2.5
    from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode as unset_fake_temporarily


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
    numel = functools.reduce(operator.mul, tensor_meta.shape)

    dtype = tensor_meta.dtype
    if dtype == torch.float32:
        elem_size = 4
    elif dtype == torch.float64:
        elem_size = 8
    elif dtype == torch.float16 or dtype == torch.bfloat16:
        elem_size = 2
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

    def offload(self, name: str, v: Argument) -> None:
        self.env_values[name] = map_aggregate(v, lambda x: self._to_cpu(x))

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
                return torch.randn(t.shape,
                                   dtype=t.dtype,
                                   device=t.device if device is None else device,
                                   layout=t.layout,
                                   requires_grad=t.requires_grad,
                                   pin_memory=t.is_pinned())
        return t

    return map_aggregate(v, lambda x: convert(x))
