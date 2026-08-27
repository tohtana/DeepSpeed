# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import operator

import torch

from deepspeed.compile.graph_edit import (RUNTIME_GRAPH_ID, clone_graph_module, generated_graph_fingerprint_details,
                                          normalized_graph, structural_fingerprint)


class ChainModule(torch.nn.Module):

    def forward(self, x):
        return torch.sigmoid(torch.relu(x))


def _runtime_graph(graph_id):
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    result = graph.call_function(operator.add, (value, graph_id))
    graph.output(result)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _call_module_graph(module):
    root = torch.nn.Module()
    root.add_module("operation", module)
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    result = graph.call_module("operation", (value, ))
    graph.output(result)
    return torch.fx.GraphModule(root, graph)


def _callable_graph(target):
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    result = graph.call_function(target, (value, ))
    graph.output(result)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _add_one(value):
    return value + 1


def _subtract_one(value):
    return value - 1


def test_structural_fingerprint_uses_topology_not_node_names():
    first = torch.fx.symbolic_trace(ChainModule())
    second = torch.fx.symbolic_trace(ChainModule())
    for index, node in enumerate(second.graph.nodes):
        node.name = f"rank_local_hint_{index}"

    assert structural_fingerprint(first) == structural_fingerprint(second)
    assert normalized_graph(first, include_hints=True) != normalized_graph(second, include_hints=True)


def test_rank_local_graph_ids_have_one_frozen_base_identity():
    rank_zero_id = 120000000001
    rank_one_id = 890000000007
    rank_zero_graph = _runtime_graph(rank_zero_id)
    rank_one_graph = _runtime_graph(rank_one_id)

    assert structural_fingerprint(rank_zero_graph, rank_zero_id) == structural_fingerprint(rank_one_graph, rank_one_id)
    assert normalized_graph(rank_zero_graph, runtime_graph_id=rank_zero_id)[1]["args"][1] == RUNTIME_GRAPH_ID
    assert normalized_graph(rank_one_graph, runtime_graph_id=rank_one_id)[1]["args"][1] == RUNTIME_GRAPH_ID


def test_graph_clone_deep_copies_nested_node_metadata():
    frozen = torch.fx.symbolic_trace(ChainModule())
    frozen_node = list(frozen.graph.nodes)[1]
    frozen_node.meta["profile"] = {"samples": [1, {"nested": [2]}]}

    first = clone_graph_module(frozen)
    second = clone_graph_module(frozen)
    first_node = list(first.graph.nodes)[1]
    second_node = list(second.graph.nodes)[1]
    first_node.meta["profile"]["samples"][1]["nested"].append(3)

    assert frozen_node.meta["profile"]["samples"][1]["nested"] == [2]
    assert second_node.meta["profile"]["samples"][1]["nested"] == [2]


def test_generated_fingerprint_is_total_for_tensor_constants():
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    tensor_constant = torch.tensor([1.0, 2.0])
    result = graph.call_function(operator.add, (value, tensor_constant))
    graph.output(result)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    first = generated_graph_fingerprint_details(gm)
    second = generated_graph_fingerprint_details(gm)

    assert first == second
    assert len(first["fingerprint"]) == 64
    assert first["opaque_fallback_count"] == 1
    assert first["opaque_fallback_types"] == ["tensor_constant"]


def test_generated_fingerprint_reports_opaque_local_callable():

    def local_target(value):
        return value + 1

    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    result = graph.call_function(local_target, (value, ))
    graph.output(result)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    details = generated_graph_fingerprint_details(gm)

    assert details["opaque_fallback_count"] == 1
    assert details["opaque_fallback_types"] == ["opaque_callable"]


def test_generated_fingerprint_includes_same_target_module_binding_type():
    relu_graph = _call_module_graph(torch.nn.ReLU())
    sigmoid_graph = _call_module_graph(torch.nn.Sigmoid())

    relu_target = next(node.target for node in relu_graph.graph.nodes if node.op == "call_module")
    sigmoid_target = next(node.target for node in sigmoid_graph.graph.nodes if node.op == "call_module")

    assert relu_target == sigmoid_target == "operation"
    assert generated_graph_fingerprint_details(relu_graph)["fingerprint"] != generated_graph_fingerprint_details(
        sigmoid_graph)["fingerprint"]


def test_generated_fingerprint_includes_same_path_callable_implementation():
    original_identity = (_subtract_one.__module__, _subtract_one.__qualname__, _subtract_one.__name__)
    try:
        _subtract_one.__module__ = _add_one.__module__
        _subtract_one.__qualname__ = _add_one.__qualname__
        _subtract_one.__name__ = _add_one.__name__
        assert (_subtract_one.__module__, _subtract_one.__qualname__) == (_add_one.__module__, _add_one.__qualname__)
        add_graph = _callable_graph(_add_one)
        subtract_graph = _callable_graph(_subtract_one)
        add_fingerprint = generated_graph_fingerprint_details(add_graph)["fingerprint"]
        subtract_fingerprint = generated_graph_fingerprint_details(subtract_graph)["fingerprint"]
    finally:
        _subtract_one.__module__, _subtract_one.__qualname__, _subtract_one.__name__ = original_identity

    assert add_fingerprint != subtract_fingerprint


def test_generated_fingerprint_includes_nested_graph_module_structure():
    identity_graph = torch.fx.Graph()
    identity_value = identity_graph.placeholder("value")
    identity_graph.output(identity_value)
    identity_module = torch.fx.GraphModule(torch.nn.Module(), identity_graph)

    negate_graph = torch.fx.Graph()
    negate_value = negate_graph.placeholder("value")
    negate_result = negate_graph.call_function(operator.neg, (negate_value, ))
    negate_graph.output(negate_result)
    negate_module = torch.fx.GraphModule(torch.nn.Module(), negate_graph)

    identity_outer = _call_module_graph(identity_module)
    negate_outer = _call_module_graph(negate_module)

    assert generated_graph_fingerprint_details(identity_outer)["fingerprint"] != generated_graph_fingerprint_details(
        negate_outer)["fingerprint"]
