# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

import torch
from torch.fx import Graph, GraphModule

from deepspeed.compile.passes import prefetch

_DC_LIBRARIES = []


def _define_dc_ops():
    try:
        torch.ops.dc.allgather_param.default
        torch.ops.dc.prefetch_params_fused.default
        torch.ops.dc.reload_parameter.default
        return
    except AttributeError:
        pass

    lib = torch.library.Library("dc", "DEF")
    for schema in (
            "allgather_param(Tensor a, int graph_id, int id, ScalarType? dtype = None) -> Tensor",
            "prefetch_params_fused(int graph_id, Tensor[] params, int[] ids) -> ()",
            "reload_parameter(Tensor(a) a, int graph_id, int id) -> Tensor(a)",
    ):
        try:
            lib.define(schema)
        except RuntimeError as exc:
            if "already been registered" not in str(exc):
                raise
    _DC_LIBRARIES.append(lib)


class _FakeAccelerator:

    def total_memory(self):
        return 1024

    def current_device(self):
        return "cpu"

    def available_memory(self):
        return 1024

    def memory_allocated(self):
        return 0

    def max_memory_allocated(self):
        return 0


class _Root(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = torch.tensor(1.0)


def _profiling_for_graph(graph):
    mem = []
    op_time = []
    tensor_sizes = []
    for node in graph.nodes:
        mem.append((node.name, 0, 0, 0))
        op_time.append((node.name, 0, 0))
        tensor_sizes.append((node.name, 1))
    return SimpleNamespace(fwd_mem=mem, fwd_time=op_time, fwd_tensor_sizes=tensor_sizes)


def test_schedule_prefetch_allows_non_placeholder_first_node(monkeypatch):
    _define_dc_ops()
    monkeypatch.setattr(prefetch, "get_accelerator", lambda: _FakeAccelerator())
    monkeypatch.setattr(prefetch.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(prefetch.dist, "all_reduce", lambda tensor, op: None)
    monkeypatch.setattr(prefetch.dist, "ReduceOp", SimpleNamespace(MIN=object()))
    monkeypatch.setattr(prefetch, "create_predictor", lambda: (lambda size: 0))

    graph = Graph()
    scale = graph.get_attr("scale")
    arg = graph.placeholder("arg")
    gathered = graph.call_function(torch.ops.dc.allgather_param.default, (arg, 0, 1), name="allgather")
    graph.output((scale, gathered))

    gm = GraphModule(_Root(), graph)
    profiling_results = {0: _profiling_for_graph(gm.graph)}

    result = prefetch.schedule_prefetch(
        gm,
        graph_id=0,
        graph_order=[(0, False)],
        profiling_results=profiling_results,
        create_inputs_fn=None,
        mem_budget=0,
        param_manager=None,
        bwd=False,
    )

    result.graph.lint()
    nodes = list(result.graph.nodes)
    prefetch_idx = next(i for i, node in enumerate(nodes) if node.target == torch.ops.dc.prefetch_params_fused.default)
    scale_idx = next(i for i, node in enumerate(nodes) if node.name == "scale")
    arg_idx = next(i for i, node in enumerate(nodes) if node.name == "arg")
    allgather_idx = next(i for i, node in enumerate(nodes) if node.name == "allgather")

    assert scale_idx < prefetch_idx
    assert arg_idx < prefetch_idx < allgather_idx
