# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import operator
from types import SimpleNamespace

import pytest
import torch
from torch.fx import Graph, GraphModule

import deepspeed.compile.util as compile_util
from deepspeed.compile.passes import offload_parameters as offload_mod

_DC_LIBRARIES = []


def _define_dc_ops():
    try:
        torch.ops.dc.offload_parameter.default
        return
    except AttributeError:
        pass

    lib = torch.library.Library("dc", "DEF")
    for schema in (
            "allgather_param(Tensor a, int graph_id, int id, ScalarType? dtype = None) -> Tensor",
            "wait_allgather(Tensor(a) a, int graph_id, int id) -> Tensor(a)",
            "offload_parameter(Tensor a, int graph_id, int id) -> ()",
            "reload_parameter(Tensor a, int graph_id, int id) -> ()",
    ):
        try:
            lib.define(schema)
        except RuntimeError as exc:
            if "already been registered" not in str(exc):
                raise
    _DC_LIBRARIES.append(lib)


@pytest.fixture(autouse=True)
def stub_deepcompile_ops(monkeypatch):
    _define_dc_ops()
    monkeypatch.setattr(compile_util, "get_no_copy_ops", lambda: {torch.ops.dc.wait_allgather.default})


def _liveout_graph():
    graph = Graph()
    param = graph.placeholder("param")
    allgather = graph.call_function(torch.ops.dc.allgather_param.default, (param, 0, 1))
    wait = graph.call_function(torch.ops.dc.wait_allgather.default, (allgather, 0, 1))
    output = graph.output(wait)
    return GraphModule({}, graph), wait, output


def _consumed_graph():
    graph = Graph()
    param = graph.placeholder("param")
    allgather = graph.call_function(torch.ops.dc.allgather_param.default, (param, 0, 1))
    wait = graph.call_function(torch.ops.dc.wait_allgather.default, (allgather, 0, 1))
    consumer = graph.call_function(operator.neg, (wait, ))
    output = graph.output(consumer)
    return GraphModule({}, graph), consumer, output


def _apply_offload(gm):
    return offload_mod.offload_parameter_fwd(gm, 0, [], None, None, 0, None, False)


def _node(gm, target):
    return next(node for node in gm.graph.nodes if node.target == target)


def test_liveout_offload_is_deferred_until_backward(monkeypatch):
    gm, wait, output = _liveout_graph()
    deferred = []
    handle = SimpleNamespace(defer_parameter_offload=lambda graph_id, ds_id: deferred.append((graph_id, ds_id)))
    monkeypatch.setattr(offload_mod, "get_deepcompile_handle", lambda: handle)

    _apply_offload(gm)

    nodes = list(gm.graph.nodes)
    assert deferred == [(0, 1)]
    assert all(node.target != torch.ops.dc.offload_parameter.default for node in nodes)
    assert nodes.index(wait) < nodes.index(output)


def test_in_graph_consumer_remains_the_offload_dependency(monkeypatch):
    gm, consumer, output = _consumed_graph()
    handle = SimpleNamespace(defer_parameter_offload=lambda *_: pytest.fail("offload should not be deferred"))
    monkeypatch.setattr(offload_mod, "get_deepcompile_handle", lambda: handle)

    _apply_offload(gm)

    offload = _node(gm, torch.ops.dc.offload_parameter.default)
    nodes = list(gm.graph.nodes)
    assert offload.args[0] is consumer
    assert nodes.index(consumer) < nodes.index(offload) < nodes.index(output)
