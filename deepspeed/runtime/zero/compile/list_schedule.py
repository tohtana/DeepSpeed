# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict
from typing import List, Dict
from copy import copy

from torch.fx import Graph, Node
from torch.utils._pytree import tree_iter

from .util import tensor_meta_size


def init_schedule(graph: Graph):
    mem_table = create_mem_table(graph)

    scheduled = []
    unscheduled = []
    edges = defaultdict(list)
    for node in graph.nodes:
        # print(f"Node: {node} args: {node.args}")
        if len(node.args) == 0:
            scheduled.append(node)
        else:
            unscheduled.append(node)
        for a in node.args:
            for elem_a in tree_iter(a):
                if isinstance(elem_a, Node):
                    if node not in edges[elem_a]:
                        edges[elem_a].append(node)

    return scheduled, unscheduled, edges, mem_table


def make_graph_from_schedule(scheduled: List[Node]):
    new_graph = Graph()
    env = {}
    for node in scheduled:
        new_node = new_graph.node_copy(node, lambda n: env[n.name])
        env[node.name] = new_node

    return new_graph


def get_original_args_num(node: Node):
    if node.name.startswith("allgather_ds_param") \
        or node.name.startswith("release_ds_param") \
        or node.name.startswith("wait_allgather_ds_param") \
        or node.name.startswith("reduce_ds_param"):
        return 1

    return len(node.args)


def flat_nodes_in_args(args: List[Node]):
    return [a for a in tree_iter(args) if isinstance(a, Node)]


def filter_args(node: Node):
    args = node.args[:get_original_args_num(node)]
    return flat_nodes_in_args(args)


def get_runnable_nodes(scheduled: List[Node], unscheduled: List[Node]):
    scheduled = set(scheduled)
    return [node for node in unscheduled if all(arg in scheduled for arg in filter_args(node))]


def choose_next_node(scheduled: List[Node], unscheduled: List[Node], mem_table: Dict[str, int]):
    runnable_nodes = get_runnable_nodes(scheduled, unscheduled)

    # sort by memory usage
    runnable_nodes = sorted(runnable_nodes, key=lambda n: mem_table[n.name])
    return runnable_nodes[0]


def create_mem_table(graph: Graph) -> Dict[str, int]:
    mem_table = {}
    for node in graph.nodes:
        if node.name.startswith("allgather_ds_param"):
            mem_table[node.name] = tensor_meta_size(node.meta["tensor_meta"])
        elif node.name.startswith("release_ds_param") or node.name.startswith("reduce_ds_param"):
            mem_table[node.name] = -tensor_meta_size(node.meta["tensor_meta"])
        else:
            mem_table[node.name] = 0

    return mem_table


def list_schedule(graph: Graph) -> Graph:

    scheduled, unscheduled, mem_table = init_schedule(graph)

    while len(unscheduled) > 0:
        next_node = choose_next_node(scheduled, unscheduled, mem_table)
        scheduled.append(next_node)
        unscheduled.remove(next_node)

    return make_graph_from_schedule(scheduled)


###############################


def get_new_runnable_nodes_with(scheduled: List[Node], edges: Dict[Node, List[Node]], new_scheduled: Node):
    scheduled = set(scheduled)
    new_runnables = []
    for node in edges[new_scheduled]:
        args = node.args[:get_original_args_num(node)]
        if all(arg in scheduled for arg in filter_args(node) if arg != new_scheduled):
            new_runnables.append(node)

    return new_runnables


def _do_schedule_without_allgather(scheduled: List[Node], unscheduled: List[Node], edges: Dict[Node, List[Node]],
                                   non_ag_runnable: List[Node]):

    while len(non_ag_runnable) > 0:
        next_node = non_ag_runnable.pop()

        new_runnables = get_new_runnable_nodes_with(scheduled, edges, next_node)
        non_ag_runnable += [n for n in new_runnables if not n.name.startswith("allgather_ds_param")]

        scheduled.append(next_node)
        unscheduled.remove(next_node)

    return scheduled, unscheduled


def schedule_without_allgather(scheduled: List[Node], unscheduled: List[Node], edges: Dict[Node, List[Node]]):
    runnable = get_runnable_nodes(scheduled, unscheduled)
    non_ag_runnable = [n for n in runnable if not n.name.startswith("allgather_ds_param")]

    tmp_scheduled = copy(scheduled)
    tmp_unscheduled = copy(unscheduled)

    return _do_schedule_without_allgather(tmp_scheduled, tmp_unscheduled, edges, non_ag_runnable)


def try_schedule_with_new_allgather(scheduled: List[Node], unscheduled: List[Node], edges: Dict[Node, List[Node]],
                                    new_scheduled: Node):
    new_runnables = get_new_runnable_nodes_with(scheduled, edges, new_scheduled)
    non_ag_runnable = [n for n in new_runnables if not n.name.startswith("allgather_ds_param")]

    tmp_scheduled = copy(scheduled)
    tmp_unscheduled = copy(unscheduled)

    tmp_scheduled.append(new_scheduled)
    tmp_unscheduled.remove(new_scheduled)

    return _do_schedule_without_allgather(tmp_scheduled, tmp_unscheduled, edges, non_ag_runnable)


def list_schedule2(graph: Graph) -> Graph:

    scheduled, unscheduled, edges, mem_table = init_schedule(graph)
    tmp_scheduled, tmp_unscheduled = schedule_without_allgather(scheduled, unscheduled, edges)

    while len(tmp_unscheduled) > 0:

        runnable = get_runnable_nodes(tmp_scheduled, tmp_unscheduled)
        ag_with_unblock_time = []

        for ag_node in runnable:
            ag_scheduled, ag_unscheduled = try_schedule_with_new_allgather(tmp_scheduled, tmp_unscheduled, edges,
                                                                           ag_node)
            unblock_time = sum(n.meta["device_time"] for n in ag_scheduled[len(tmp_scheduled) + 1:])
            ag_with_unblock_time.append((ag_node, unblock_time, ag_scheduled, ag_unscheduled))

        ag_with_unblock_time = sorted(ag_with_unblock_time, key=lambda x: x[1], reverse=True)
        best_ag_node = ag_with_unblock_time[0][0]
        best_ag_scheduled = ag_with_unblock_time[0][2]

        no_ag_runnables = tmp_scheduled[len(scheduled):]
        after_ag_runnables = best_ag_scheduled[len(tmp_scheduled) + 1:]

        scheduled.append(best_ag_node)
        unscheduled.remove(best_ag_node)
        for n in no_ag_runnables:
            scheduled.append(n)
            unscheduled.remove(n)

        tmp_scheduled = copy(scheduled)
        tmp_unscheduled = copy(unscheduled)
        for n in after_ag_runnables:
            tmp_scheduled.append(n)
            tmp_unscheduled.remove(n)

    return make_graph_from_schedule(tmp_scheduled)
