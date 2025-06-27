import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree


def solve_tsp_christofides(cities):
    """
    A scalable implementation of the Christofides algorithm that uses a
    fast, greedy algorithm for the minimum-weight perfect matching step.
    This version can handle larger problems like 30 cities.
    """
    num_cities = len(cities)
    if num_cities == 0:
        return [], 0.0

    dist_matrix = squareform(pdist(cities, 'euclidean'))

    # 1. Create a Minimum Spanning Tree (MST)
    mst = minimum_spanning_tree(dist_matrix)

    # 2. Find vertices with odd degree in the MST
    mst_adj = mst.toarray()
    degrees = np.sum(mst_adj > 0, axis=1) + np.sum(mst_adj > 0, axis=0)
    odd_degree_nodes = np.where(degrees % 2 != 0)[0]

    # 3. Find a minimum-weight perfect matching using a greedy algorithm
    final_matching_pairs = greedy_matching(odd_degree_nodes, dist_matrix)

    # 4. Combine MST and matching to form an Eulerian multigraph
    multigraph = [[] for _ in range(num_cities)]
    mst_rows, mst_cols = mst.nonzero()
    for u, v in zip(mst_rows, mst_cols):
        multigraph[u].append(v)
        multigraph[v].append(u)

    for u, v in final_matching_pairs:
        multigraph[u].append(v)
        multigraph[v].append(u)

    # 5. Find an Eulerian circuit
    start_node = 0
    eulerian_circuit = find_eulerian_circuit(multigraph, start_node)

    # 6. Convert to Hamiltonian circuit (final tour) using greedy shortcutting
    final_tour = []
    visited = set()
    for node in eulerian_circuit:
        if node not in visited:
            final_tour.append(node)
            visited.add(node)
    final_tour.append(final_tour[0])

    # 7. Calculate final tour length
    tour_length = 0
    for i in range(len(final_tour) - 1):
        tour_length += dist_matrix[final_tour[i], final_tour[i + 1]]

    return final_tour, tour_length, eulerian_circuit


def greedy_matching(odd_nodes, dist_matrix):
    """
    A fast, greedy algorithm for the minimum-weight perfect matching.
    """
    nodes = list(odd_nodes)
    pairs = []

    while nodes:
        u = nodes.pop(0)
        closest_dist = float('inf')
        closest_v = -1

        for v in nodes:
            if dist_matrix[u, v] < closest_dist:
                closest_dist = dist_matrix[u, v]
                closest_v = v

        if closest_v != -1:
            pairs.append(tuple(sorted((u, closest_v))))
            nodes.remove(closest_v)

    return pairs


def find_eulerian_circuit(multigraph, start_node):
    """Finds an Eulerian circuit using Hierholzer's algorithm."""
    if not multigraph[start_node]:
        return [start_node]

    temp_graph = [list(neighbors) for neighbors in multigraph]
    stack = [start_node]
    circuit = []

    while stack:
        u = stack[-1]
        if temp_graph[u]:
            v = temp_graph[u].pop(0)
            temp_graph[v].remove(u)
            stack.append(v)
        else:
            circuit.append(stack.pop())

    return circuit[::-1]