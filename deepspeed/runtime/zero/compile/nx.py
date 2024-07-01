import torch.fx as fx
import networkx as nx

def fx_to_nx(fx_graph):
    """Converts a torch.fx.Graph to a NetworkX graph."""
    nx_graph = nx.DiGraph()

    def traverse(start_node: fx.Node):
        stack = [start_node]
        while stack:
            node = stack.pop()
            if node in nx_graph:
                continue
            nx_graph.add_node(node)
            for next_node in node.users.keys():
                if next_node not in nx_graph:
                    stack.append(next_node)
                nx_graph.add_edge(node, next_node)

    for node in fx_graph.nodes:
        traverse(node)

    # A = nx.nx_agraph.to_agraph(nx_graph)
    # A.draw("graph.png", format="png", prog="dot")
    
    return nx_graph


def find_reachable_terminal_nodes(graph, marked_nodes):
    """
    Find marked nodes in a directed graph that can reach the terminal node(s) without passing through other marked nodes.

    Parameters:
    - graph: NetworkX DiGraph object, the directed graph.
    - marked_nodes: List of nodes marked with the initial mark.

    Returns:
    - List of marked nodes that can reach the terminal node(s) without passing through other marked nodes.
    """
    graph = graph.copy()  # Avoid modifying the original graph
    for marked_node in marked_nodes:
        to_remove = []
        for _, dest in graph.out_edges(marked_node):
            if len(graph.out_edges(dest)) == 0:
                to_remove.append((marked_node, dest))
        graph.remove_edges_from(to_remove)

    terminal_nodes = [node for node in graph.nodes if graph.out_degree(node) == 0]
    reachable_marked_nodes = []

    for marked_node in marked_nodes:
        visited = set()
        queue = [(marked_node, False)]  # (current_node, passed_through_marked_node)
        
        while queue:
            current_node, passed_through_marked_node = queue.pop(0)
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            if current_node in terminal_nodes and not passed_through_marked_node:
                reachable_marked_nodes.append(marked_node)
                break

            for neighbor in graph.successors(current_node):
                if neighbor in marked_nodes and neighbor != marked_node:
                    continue
                queue.append((neighbor, passed_through_marked_node or (neighbor in marked_nodes and neighbor != marked_node)))

    return reachable_marked_nodes