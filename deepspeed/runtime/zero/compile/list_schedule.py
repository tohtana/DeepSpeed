# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict
from typing import List, Dict, Set
from copy import copy

from torch.fx import Graph, Node
from torch.utils._pytree import tree_iter

from .util import tensor_meta_size

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


def init_schedule(graph: Graph):
    mem_table = create_mem_table(graph)
    remaining_users = defaultdict(set)
    user_to_producer = {}

    scheduled = []
    unscheduled = []
    edges = defaultdict(list)
    for node in graph.nodes:
        filtered_args = filter_args(node)
        # print(f"Node: {node} args: {node.args}")
        if len(filtered_args) == 0:
            scheduled.append(node)

            remaining_users[node] = set(node.users.keys())
            for user in node.users.keys():
                user_to_producer[user] = node
        else:
            unscheduled.append(node)
        for a in filtered_args:
            for elem_a in tree_iter(a):
                if isinstance(elem_a, Node):
                    if node not in edges[elem_a]:
                        edges[elem_a].append(node)

    return scheduled, unscheduled, edges, mem_table, remaining_users, user_to_producer


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
                                   non_ag_runnable: List[Node],
                                   remaining_users: Dict[Node, Set[Node]], user_to_producer: Dict[Node, Node]):
    # scheduled, unscheduled, and remaining_users are modified in place

    print(f"_do_schedule_without_allgather: scheduled: {scheduled} unscheduled: {unscheduled} non_ag_runnable: {non_ag_runnable} remaining_users: {remaining_users} user_to_producer: {user_to_producer}")
    # inflightを洗い出す
    # ここでは、scheduledのうち、userがあるものがinflightになる、基本的には全部あるはず
    # 次に、実行の候補としては、non_ag_runnableである
    # 何らかの形でinflightのuserであるわけだが、実行したときに、何かがinflightでなくなる可能性がある、そのサイズを合算する
    # これを求めるために、scheduledのそれぞれで、クリアするための残りのノードを保存しておく -> remaining_users

    while len(non_ag_runnable) > 0:
        # for n in non_ag_runnable:
        #     print(f"  check users Node: {n.name} users: {list(n.users.keys())} producer: {user_to_producer[n]}")
        #     producer = user_to_producer[n]
        #     remaining_users_of_producer = remaining_users[producer]
        #     if len(remaining_users_of_producer) == 1 and next(iter(remaining_users_of_producer)) == n:
        #         print(f"Node: {n.name} is the last user of {producer.name}. size: {n.meta['tensor_size']}")

        next_node = non_ag_runnable.pop()

        new_runnables = get_new_runnable_nodes_with(scheduled, edges, next_node)
        non_ag_runnable += [n for n in new_runnables if not n.name.startswith("allgather_ds_param")]

        scheduled.append(next_node)
        unscheduled.remove(next_node)

        # producer = user_to_producer[next_node]
        # remaining_users[producer].remove(next_node)

    # return scheduled, unscheduled


def schedule_without_allgather(scheduled: List[Node], unscheduled: List[Node], edges: Dict[Node, List[Node]],
                               remaining_users: Dict[Node, Set[Node]], user_to_producer: Dict[Node, Node]):
    runnable = get_runnable_nodes(scheduled, unscheduled)
    non_ag_runnable = [n for n in runnable if not n.name.startswith("allgather_ds_param")]

    tmp_scheduled = copy(scheduled)
    tmp_unscheduled = copy(unscheduled)
    tmp_remaining_users = copy(remaining_users)

    _do_schedule_without_allgather(tmp_scheduled, tmp_unscheduled, edges, non_ag_runnable, tmp_remaining_users, user_to_producer)
    return tmp_scheduled, tmp_unscheduled, tmp_remaining_users


def try_schedule_with_new_allgather(scheduled: List[Node], unscheduled: List[Node], edges: Dict[Node, List[Node]],
                                    new_scheduled: Node, remaining_users: Dict[Node, Set[Node]], user_to_producer: Dict[Node, Node]):
    new_runnables = get_new_runnable_nodes_with(scheduled, edges, new_scheduled)
    non_ag_runnable = [n for n in new_runnables if not n.name.startswith("allgather_ds_param")]

    tmp_scheduled = copy(scheduled)
    tmp_unscheduled = copy(unscheduled)
    tmp_remaining_users = copy(remaining_users)

    tmp_scheduled.append(new_scheduled)
    tmp_unscheduled.remove(new_scheduled)

    _do_schedule_without_allgather(tmp_scheduled, tmp_unscheduled, edges, non_ag_runnable, tmp_remaining_users, user_to_producer)
    return tmp_scheduled, tmp_unscheduled, tmp_remaining_users


def count_inflight_values(graph: Graph):

    node_to_last_use: Dict[Node, Node] = {}
    user_to_last_uses: Dict[Node, List[Node]] = {}

    def register_last_uses(n: Node, user: Node):
        if n not in node_to_last_use:
            node_to_last_use[n] = user
            user_to_last_uses.setdefault(user, []).append(n)

    from torch.fx.node import map_arg
    for node in reversed(graph.nodes):
        map_arg(node.args, lambda n: register_last_uses(n, node))
        map_arg(node.kwargs, lambda n: register_last_uses(n, node))

    max_inflight_size = 0
    inflight_values = set()
    for node in graph.nodes:
        inflight_values.add(node)
        if node in user_to_last_uses:
            for to_delete in user_to_last_uses[node]:
                inflight_values.remove(to_delete)

        inflight_size = sum(n.meta["tensor_size"] for n in inflight_values)
        print(
            f"Node: {node.name} users: {list(node.users.keys())} node_to_last_use: {node_to_last_use[node] if node in node_to_last_use else 'NA'} user_to_last_uses: {user_to_last_uses[node] if node in user_to_last_uses else 'NA'} inflight_values: {inflight_values} inflight_size: {inflight_size}"
        )
        max_inflight_size = max(max_inflight_size, inflight_size)
    print(f"Max inflight size: {max_inflight_size}")


def list_schedule2(graph: Graph, available_mem: int, output_size: int, debug_log: bool) -> Graph:

    scheduled, unscheduled, edges, mem_table, remaining_users, user_to_producer = init_schedule(graph)
    tmp_scheduled, tmp_unscheduled, tmp_remaining_users = schedule_without_allgather(scheduled, unscheduled, edges, remaining_users, user_to_producer)

    while len(tmp_unscheduled) > 0:

        runnable = get_runnable_nodes(tmp_scheduled, tmp_unscheduled)
        ag_with_unblock_time = []

        for ag_node in runnable:
            ag_scheduled, ag_unscheduled, ag_remaining_users = try_schedule_with_new_allgather(tmp_scheduled, tmp_unscheduled, edges,
                                                                           ag_node, tmp_remaining_users, user_to_producer)
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

    ret = make_graph_from_schedule(tmp_scheduled)
    if debug_log:
        count_inflight_values(ret)
    return ret
