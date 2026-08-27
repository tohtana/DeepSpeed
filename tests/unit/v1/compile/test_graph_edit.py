# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import copy
from dataclasses import asdict
import json
import operator
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import deepspeed.compile.optimizer as optimizer_module
from deepspeed.compile.evaluation_context import (EVALUATION_SCHEMA_VERSION, AgentResponseError, GraphSlotRef,
                                                  GraphVersionTracker, parse_evaluation_decision,
                                                  serialize_evaluation_context)
from deepspeed.compile.graph_edit import (RUNTIME_GRAPH_ID, GraphEditError, GraphEditPayload, apply_graph_edit,
                                          candidate_fingerprint, finalize_graph_edit, normalized_graph,
                                          structural_fingerprint)


class RichModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.block = torch.nn.ReLU()
        self.weight = torch.nn.Parameter(torch.ones(3))

    def forward(self, x):
        return self.block(x) + self.weight


class ChainModule(torch.nn.Module):

    def forward(self, x):
        relu = torch.relu(x)
        return torch.sigmoid(relu)


def _rich_replacement_log(gm):
    operations = [
        {
            "op": "create_node",
            "id": "new:x",
            "node_op": "placeholder",
            "target": "x",
            "args": [],
            "kwargs": {},
            "name_hint": "fresh_x",
        },
        {
            "op": "create_node",
            "id": "new:weight",
            "node_op": "get_attr",
            "target": "weight",
            "args": [],
            "kwargs": {},
        },
        {
            "op": "create_node",
            "id": "new:block",
            "node_op": "call_module",
            "target": "block",
            "args": [{
                "node": "new:x"
            }],
            "kwargs": {},
        },
        {
            "op": "create_node",
            "id": "new:add",
            "node_op": "call_method",
            "target": "add",
            "args": [{
                "node": "new:block"
            }, {
                "node": "new:weight"
            }],
            "kwargs": {},
        },
        {
            "op": "create_node",
            "id": "new:slice",
            "node_op": "call_function",
            "target": "operator.getitem",
            "args": [{
                "node": "new:add"
            }, {
                "slice": [0, None, None]
            }],
            "kwargs": {},
        },
        {
            "op":
            "create_node",
            "id":
            "new:output",
            "node_op":
            "output",
            "target":
            "output",
            "args": [{
                "dict": [["value", {
                    "node": "new:slice"
                }], ["nested", {
                    "tuple": [[1, 2], {
                        "slice": [None, None, None]
                    }]
                }]]
            }],
            "kwargs": {},
        },
    ]
    operations.extend({"op": "delete_node", "id": f"base:{index}"} for index in range(4, -1, -1))
    operations.append({
        "op": "reorder",
        "order": ["new:x", "new:weight", "new:block", "new:add", "new:slice", "new:output"],
    })
    return GraphEditPayload(generation=1,
                            graph_slot=(0, "fwd"),
                            base_fingerprint=structural_fingerprint(gm),
                            expected_result_fingerprint=None,
                            operations=operations,
                            reason="Exercise every FX node kind and nested argument encoding")


def test_edit_log_creates_every_fx_node_kind_with_nested_arguments():
    gm = torch.fx.symbolic_trace(RichModule())
    payload, candidate = finalize_graph_edit(gm, _rich_replacement_log(gm))

    candidate.graph.lint()
    result = candidate(torch.tensor([2.0, 3.0, 4.0]))
    assert torch.equal(result["value"], torch.tensor([3.0, 4.0, 5.0]))
    assert result["nested"] == ([1, 2], slice(None, None, None))
    assert [node.op for node in candidate.graph.nodes
            ] == ["placeholder", "get_attr", "call_module", "call_method", "call_function", "output"]
    assert candidate.weight is gm.weight
    assert payload.expected_result_fingerprint == candidate_fingerprint(candidate, payload)
    assert payload.expected_result_fingerprint != structural_fingerprint(candidate)


def test_edit_log_rewires_deletes_reorders_and_copies_only_local_metadata():
    gm = torch.fx.symbolic_trace(ChainModule())
    relu = list(gm.graph.nodes)[1]
    relu.meta["rank_local"] = object()
    operations = [{
        "op": "create_node",
        "id": "new:neg",
        "node_op": "call_function",
        "target": "torch.neg",
        "args": [{
            "node": "base:1"
        }],
        "kwargs": {},
        "copy_meta_from": "base:1",
        "meta": {
            "device_time": 0.0
        },
    }, {
        "op": "rewire",
        "id": "base:3",
        "args": [{
            "node": "new:neg"
        }],
    }, {
        "op": "delete_node",
        "id": "base:2",
    }, {
        "op": "reorder",
        "order": ["base:0", "base:1", "new:neg", "base:3"],
    }]
    raw = GraphEditPayload(generation=1,
                           graph_slot=(0, "fwd"),
                           base_fingerprint=structural_fingerprint(gm),
                           expected_result_fingerprint=None,
                           operations=operations)
    payload, candidate = finalize_graph_edit(gm, raw)
    replayed = apply_graph_edit(gm, payload)

    assert torch.equal(candidate(torch.tensor([-2.0, 3.0])), torch.tensor([0.0, -3.0]))
    assert structural_fingerprint(replayed) == structural_fingerprint(candidate)
    new_node = next(node for node in candidate.graph.nodes if node.target == torch.neg)
    assert "rank_local" in new_node.meta
    assert new_node.meta["device_time"] == 0.0
    assert next(node for node in gm.graph.nodes if node.target == torch.sigmoid)


def test_edit_log_rejects_only_mechanical_serialization_and_graph_failures():
    gm = torch.fx.symbolic_trace(ChainModule())
    payload = _rich_replacement_log(torch.fx.symbolic_trace(RichModule()))
    payload.base_fingerprint = structural_fingerprint(gm)
    payload.operations = [{
        "op": "create_node",
        "id": "new:closure",
        "node_op": "call_function",
        "target": "missing.module.closure",
        "args": [],
        "kwargs": {},
    }, {
        "op": "reorder",
        "order": ["base:0", "base:1", "base:2", "new:closure", "base:3"],
    }]

    with pytest.raises(GraphEditError, match="import|resolve"):
        finalize_graph_edit(gm, payload)


def test_nested_data_only_meta_is_deep_copied_for_create_and_patch_replay():
    gm = torch.fx.symbolic_trace(ChainModule())
    created_meta = {"schedule": {"shape": [1, 2], "labels": ["x", None]}}
    patched_meta = {"planner": {"buckets": [[1, 2], [3]], "enabled": True}}
    raw = GraphEditPayload(generation=1,
                           graph_slot=(0, "fwd"),
                           base_fingerprint=structural_fingerprint(gm),
                           expected_result_fingerprint=None,
                           operations=[{
                               "op": "patch_meta",
                               "id": "base:1",
                               "meta": patched_meta,
                           }, {
                               "op": "create_node",
                               "id": "new:neg",
                               "node_op": "call_function",
                               "target": "torch.neg",
                               "args": [{
                                   "node": "base:1"
                               }],
                               "kwargs": {},
                               "meta": created_meta,
                           }, {
                               "op": "reorder",
                               "order": ["base:0", "base:1", "base:2", "new:neg", "base:3"],
                           }])

    payload, candidate = finalize_graph_edit(gm, raw)
    replayed = apply_graph_edit(gm, payload)
    created_meta["schedule"]["shape"][0] = 99
    patched_meta["planner"]["buckets"][0][0] = 99

    candidate_nodes = list(candidate.graph.nodes)
    replayed_nodes = list(replayed.graph.nodes)
    assert candidate_nodes[1].meta["planner"]["buckets"] == [[1, 2], [3]]
    assert replayed_nodes[1].meta["planner"]["buckets"] == [[1, 2], [3]]
    assert candidate_nodes[3].meta["schedule"]["shape"] == [1, 2]
    assert replayed_nodes[3].meta["schedule"]["shape"] == [1, 2]


@pytest.mark.parametrize("invalid_meta", [(1, 2), torch.tensor([1]), torch.neg])
def test_meta_patch_rejects_non_json_values(invalid_meta):
    gm = torch.fx.symbolic_trace(ChainModule())
    raw = GraphEditPayload(generation=1,
                           graph_slot=(0, "fwd"),
                           base_fingerprint=structural_fingerprint(gm),
                           expected_result_fingerprint=None,
                           operations=[{
                               "op": "patch_meta",
                               "id": "base:1",
                               "meta": {
                                   "invalid": invalid_meta
                               },
                           }, {
                               "op": "reorder",
                               "order": ["base:0", "base:1", "base:2", "base:3"],
                           }])

    with pytest.raises(GraphEditError, match="JSON/data-only"):
        finalize_graph_edit(gm, raw)


def test_metadata_only_edits_have_distinct_candidate_fingerprints():
    gm = torch.fx.symbolic_trace(ChainModule())

    def finalize(value):
        raw = GraphEditPayload(generation=1,
                               graph_slot=(0, "fwd"),
                               base_fingerprint=structural_fingerprint(gm),
                               expected_result_fingerprint=None,
                               operations=[{
                                   "op": "patch_meta",
                                   "id": "base:1",
                                   "meta": {
                                       "probe": {
                                           "value": [value]
                                       }
                                   },
                               }, {
                                   "op": "reorder",
                                   "order": ["base:0", "base:1", "base:2", "base:3"],
                               }])
        return finalize_graph_edit(gm, raw)

    first_payload, first_candidate = finalize(1)
    second_payload, second_candidate = finalize(2)

    assert structural_fingerprint(first_candidate) == structural_fingerprint(second_candidate)
    assert first_payload.expected_result_fingerprint != second_payload.expected_result_fingerprint
    assert candidate_fingerprint(first_candidate, first_payload) == first_payload.expected_result_fingerprint
    assert candidate_fingerprint(second_candidate, second_payload) == second_payload.expected_result_fingerprint


class FakeBroadcastDist:

    def __init__(self):
        self.rank = 0
        self.messages = []
        self.read_index = 0

    def is_initialized(self):
        return True

    def get_world_size(self):
        return 2

    def get_rank(self):
        return self.rank

    def broadcast_object_list(self, object_list, src):
        if self.rank == src:
            self.messages.append(copy.deepcopy(object_list[0]))
        else:
            object_list[0] = copy.deepcopy(self.messages[self.read_index])
            self.read_index += 1


def test_rank_zero_broadcasts_complete_json_log_and_other_rank_replays(monkeypatch):
    rank_zero_graph = torch.fx.symbolic_trace(ChainModule())
    other_rank_graph = torch.fx.symbolic_trace(ChainModule())
    for index, node in enumerate(other_rank_graph.graph.nodes):
        node.name = f"other_rank_hint_{index}"
    raw = GraphEditPayload(generation=1,
                           graph_slot=(0, "fwd"),
                           base_fingerprint=structural_fingerprint(rank_zero_graph),
                           expected_result_fingerprint=None,
                           operations=[{
                               "op": "rewire",
                               "id": "base:3",
                               "args": [{
                                   "node": "base:1"
                               }],
                           }, {
                               "op": "delete_node",
                               "id": "base:2",
                           }, {
                               "op": "reorder",
                               "order": ["base:0", "base:1", "base:3"],
                           }])
    finalized, rank_zero_candidate = finalize_graph_edit(rank_zero_graph, raw)
    fake_dist = FakeBroadcastDist()
    monkeypatch.setattr(optimizer_module, "dist", fake_dist)

    received_on_rank_zero = optimizer_module.broadcast_edit_payload(finalized)
    fake_dist.rank = 1
    received_on_other_rank = optimizer_module.broadcast_edit_payload(None)
    other_candidate = apply_graph_edit(other_rank_graph, received_on_other_rank)

    assert received_on_rank_zero.to_dict() == received_on_other_rank.to_dict()
    assert structural_fingerprint(other_candidate) == structural_fingerprint(rank_zero_candidate)
    assert fake_dist.messages and isinstance(fake_dist.messages[0], str)


def test_structural_fingerprint_uses_topology_not_node_names():
    first = torch.fx.symbolic_trace(ChainModule())
    second = torch.fx.symbolic_trace(ChainModule())
    for index, node in enumerate(second.graph.nodes):
        node.name = f"rank_local_hint_{index}"

    assert structural_fingerprint(first) == structural_fingerprint(second)
    assert normalized_graph(first, include_hints=True) != normalized_graph(second, include_hints=True)


def test_opaque_local_callable_fingerprint_is_stable_across_independent_processes():
    script = """
import torch
from deepspeed.compile.graph_edit import structural_fingerprint

def build_graph():
    def local_target(value):
        return value + 1

    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    result = graph.call_function(local_target, (value,))
    graph.output(result)
    return torch.fx.GraphModule(torch.nn.Module(), graph)

print(structural_fingerprint(build_graph()))
"""
    fingerprints = []
    for _ in range(2):
        output = subprocess.check_output([sys.executable, "-c", script], text=True)
        fingerprints.append(output.strip().splitlines()[-1])

    assert fingerprints[0] == fingerprints[1]


def _runtime_graph(graph_id):
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    local = graph.call_function(operator.add, args=(value, graph_id))
    graph.output(local)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def test_two_rank_graph_ids_are_canonical_and_new_nodes_replay_each_local_id():
    rank_zero_id = 120000000001
    rank_one_id = 890000000007
    rank_zero_graph = _runtime_graph(rank_zero_id)
    rank_one_graph = _runtime_graph(rank_one_id)

    rank_zero_fingerprint = structural_fingerprint(rank_zero_graph, rank_zero_id)
    assert rank_zero_fingerprint == structural_fingerprint(rank_one_graph, rank_one_id)
    assert normalized_graph(rank_zero_graph, runtime_graph_id=rank_zero_id)[1]["args"][1] == RUNTIME_GRAPH_ID
    assert normalized_graph(rank_one_graph, runtime_graph_id=rank_one_id)[1]["args"][1] == RUNTIME_GRAPH_ID

    raw = GraphEditPayload(generation=1,
                           graph_slot=(0, "fwd"),
                           base_fingerprint=rank_zero_fingerprint,
                           expected_result_fingerprint=None,
                           operations=[{
                               "op": "create_node",
                               "id": "new:local_mul",
                               "node_op": "call_function",
                               "target": "operator.mul",
                               "args": [{
                                   "node": "base:1"
                               }, dict(RUNTIME_GRAPH_ID)],
                               "kwargs": {},
                           }, {
                               "op": "rewire",
                               "id": "base:2",
                               "args": [{
                                   "node": "new:local_mul"
                               }],
                           }, {
                               "op": "reorder",
                               "order": ["base:0", "base:1", "new:local_mul", "base:2"],
                           }])
    payload, rank_zero_candidate = finalize_graph_edit(rank_zero_graph, raw, rank_zero_id)
    rank_one_candidate = apply_graph_edit(rank_one_graph, payload, rank_one_id)

    rank_zero_new = next(node for node in rank_zero_candidate.graph.nodes if node.target is operator.mul)
    rank_one_new = next(node for node in rank_one_candidate.graph.nodes if node.target is operator.mul)
    assert rank_zero_new.args[1] == rank_zero_id
    assert rank_one_new.args[1] == rank_one_id
    assert structural_fingerprint(rank_zero_candidate,
                                  rank_zero_id) == structural_fingerprint(rank_one_candidate, rank_one_id)


def test_graph_agent_edit_prompt_is_identical_for_two_rank_local_graph_ids():
    prompts = []
    for graph_id in (120000000001, 890000000007):
        gm = _runtime_graph(graph_id)
        tracker = GraphVersionTracker(GraphSlotRef(index=0, direction="fwd"), gm, graph_id)
        profile = SimpleNamespace(fwd_time=[], fwd_mem=[], fwd_tensor_sizes=[])
        ctx = SimpleNamespace(gm=gm,
                              graph_id=graph_id,
                              graph_slot=(0, "fwd"),
                              graph_order=[(graph_id, False)],
                              profiling_results={graph_id: profile},
                              bwd=False)
        prompts.append(json.loads(serialize_evaluation_context(ctx, tracker, [])))

    assert prompts[0] == prompts[1]
    assert prompts[0]["graph_runtime"]["graph_id"] == RUNTIME_GRAPH_ID
    assert prompts[0]["accepted_graph"]["nodes"][1]["args"][1] == RUNTIME_GRAPH_ID


def test_graph_agent_decision_parses_exact_edit_and_validates_identity():
    gm = torch.fx.symbolic_trace(ChainModule())
    tracker = GraphVersionTracker(GraphSlotRef(index=0, direction="fwd"), gm, 11)
    snapshot = tracker.current_ref()
    raw_edit = GraphEditPayload(generation=1,
                                graph_slot=(0, "fwd"),
                                base_fingerprint=snapshot.graph_fingerprint,
                                expected_result_fingerprint=None,
                                operations=[{
                                    "op": "reorder",
                                    "order": ["base:0", "base:1", "base:2", "base:3"],
                                }],
                                reason="Preserve the exact data-only edit")
    response = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "based_on": asdict(snapshot),
        "decision": "continue",
        "summary": "Try the exact edit",
        "graph_edit": raw_edit.to_dict(),
        "candidate_generation": None,
        "candidate_fingerprint": None,
    }

    decision = parse_evaluation_decision(json.dumps(response), snapshot, "accepted_graph")

    assert decision.graph_edit is not None
    assert decision.graph_edit.to_dict() == raw_edit.to_dict()

    invalid_edits = []
    for field, invalid_value in (("graph_slot", [1, "fwd"]), ("generation", 2), ("base_fingerprint", "stale-base"),
                                 ("expected_result_fingerprint", "premature-result")):
        invalid = copy.deepcopy(response)
        invalid["graph_edit"][field] = invalid_value
        invalid_edits.append(invalid)
    for invalid in invalid_edits:
        with pytest.raises(AgentResponseError):
            parse_evaluation_decision(json.dumps(invalid), snapshot, "accepted_graph")
