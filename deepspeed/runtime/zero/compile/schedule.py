import torch
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
from typing import Optional

import torch.fx

from .nx import fx_to_nx, nx_to_fx
from .fx import get_output_node
from .graph_param import DSGraphParamManager


def get_input_nodes(G: nx.DiGraph):
    return [n for n in G.nodes if hasattr(n, "op") and n.op == "placeholder"]


def move_input_nodes_to_front(nodes, input_nodes):
    other_nodes = [n for n in nodes if n not in input_nodes]
    return input_nodes + other_nodes


def get_nx_output_node(G: nx.DiGraph):
    for node in G.nodes:
        if node.op == "output":
            return node
    raise ValueError("No output node found")


def find_release_nodes(G: nx.DiGraph, bw=False):
    node_name = "reduce_grad" if bw else "release_param"

    release_nodes = []
    for node in G.nodes:
        if node.name.startswith(node_name):
            release_nodes.append(node)
    return release_nodes


def find_all_dependency_nodes(G: nx.DiGraph, node, dependency_graph=None) -> nx.DiGraph:
    if dependency_graph is None:
        dependency_graph = nx.DiGraph()
    
    # add node to dependency graph if not already present
    if node in dependency_graph.nodes:
        return
    
    dependency_graph.add_node(node)
    for pred in G.predecessors(node):
        find_all_dependency_nodes(G, pred, dependency_graph)
        dependency_graph.add_edge(pred, node)
        
    return dependency_graph


def sum_allgather_sizes(G: nx.DiGraph, param_manager: DSGraphParamManager, bw=False) -> int:
    allgather_sizes = 0
    for node in G.nodes:
        if param_manager.is_allgather_node(node, bw=bw):
            param_name = param_manager.allgather_param_name(node, bw=bw)
            graph_param = param_manager.get_graph_param(param_name)
            allgather_sizes += graph_param.numel
    return allgather_sizes


def schedule_by_distance(G: nx.DiGraph) -> nx.DiGraph:
    unscheduled = list(G.nodes)
    scheduled = []

    # schedule placeholder first
    for node in unscheduled:
        if node.op == "placeholder":
            scheduled.append(node)
            unscheduled.remove(node)

    print("unscheduled", unscheduled)
    print("scheduled", scheduled)

    output_node = get_output_node(G)
    distances = nx.single_source_shortest_path_length(G.reverse(), output_node)

    i = 0
    while len(unscheduled) > 0:

        schedulable = []
        for node in unscheduled:
            if all([pred in scheduled for pred in G.predecessors(node)]):
                schedulable.append(node)

        print(f"{i}: schedulable {schedulable}")

        next_node = None
        for node in schedulable:
            if node.name.startswith("release_param"):
                print(f"Chose release node {node}")
                next_node = node
            # elif node.name.startswith("allgather_param"):
            #     print(f"Chose gather node {node}")
            #     next_node = node
        if next_node is None:
            # sort schedulable nodes by distance to output
            schedulable = sorted(schedulable, key=lambda node: distances[node], reverse=True)
            next_node = schedulable[0]

        scheduled.append(next_node)
        unscheduled.remove(next_node)
        schedulable.remove(next_node)

        print(f"{i}: selected {next_node}")

        # for node in schedulable:
        #     scheduled.append(node)
        #     unscheduled.remove(node)

        print(f"{i}: unscheduled {unscheduled}")
        print(f"{i}: scheduled {scheduled}")

        i += 1


def find_allgather_graph(G: nx.DiGraph, param_manager: DSGraphParamManager, bw=False) -> Optional[nx.DiGraph]:
    release_nodes = find_release_nodes(G, bw=bw)
    if len(release_nodes) == 0:
        return None

    release_nodes_with_ag_sizes = []
    for n in find_release_nodes(G, bw=bw):
        dependencies = find_all_dependency_nodes(G, n)

        allgather_nodes = [n for n in dependencies.nodes if param_manager.is_allgather_node(n, bw=bw)]
        release_nodes_with_ag_sizes.append((n, dependencies, allgather_nodes, sum_allgather_sizes(dependencies, param_manager)))
    # sort release nodes by allgather size
    return sorted(release_nodes_with_ag_sizes, key=lambda x: x[3], reverse=False)


def sort_nodes_by_dfs(G: nx.DiGraph) -> nx.DiGraph:
    copy_G = G.copy()
    start_node = object() # Dummy node
    copy_G.add_node(start_node)
    for n in get_input_nodes(copy_G):
        copy_G.add_edge(start_node, n)

    open_nodes = [start_node]
    seen = set()
    sorted_nodes = []

    while len(open_nodes) > 0:
        node = open_nodes.pop()
        
        sorted_nodes.append(node)
        seen.add(node)

        for succ in copy_G.successors(node):
            if succ in seen:
                continue
            # check if all predecessors are in seen
            if all([pred in seen for pred in copy_G.predecessors(succ)]):
                open_nodes.append(succ)

    # drop start node
    return sorted_nodes[1:]


def schedule_by_allgather_size(G: nx.DiGraph, param_manager: DSGraphParamManager, bw=False) -> nx.DiGraph:
    
    allgather_subgraphs = find_allgather_graph(G, param_manager, bw=bw)

    i = 0
    new_graph_nodes = []
    for subgraph_info in allgather_subgraphs:
        node, subgraph, allgather_nodes, allgather_size = subgraph_info

        sorted_nodes = move_input_nodes_to_front(sort_nodes_by_dfs(subgraph), param_manager.get_input_nodes(bw=bw))
        new_graph_nodes.extend(sorted_nodes)
        i += 1

    new_graph_nodes = move_input_nodes_to_front(new_graph_nodes, param_manager.get_input_nodes(bw=bw))

    new_G = nx.DiGraph()
    for n in new_graph_nodes:
        new_G.add_node(n)
        for pred in G.predecessors(n):
            new_G.add_edge(pred, n)

    # add missing nodes in topological order
    for n in nx.topological_sort(G):
        if n not in new_G.nodes:
            new_G.add_node(n)
            for pred in G.predecessors(n):
                new_G.add_edge(pred, n)

    # to_pydot(new_G).write_svg(f"subgraph_final.svg")

    return new_G


def schedule(fx_graph: torch.fx.Graph, param_manager: DSGraphParamManager, bw=False) -> torch.fx.Graph:
    nx_graph = fx_to_nx(fx_graph)
    new_graph = schedule_by_allgather_size(nx_graph, param_manager, bw=bw)
    return nx_to_fx(new_graph)
