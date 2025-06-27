import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt
import itertools


def solve_tsp_christofides(cities):
    """
    Solves the TSP using the Christofides algorithm.
    """
    # Step 1: Create the distance matrix
    dist_matrix = calculate_distance_matrix(cities)

    # Step 2: Find the Minimum Spanning Tree (MST)
    mst = minimum_spanning_tree(dist_matrix)

    # Step 3: Find the vertices with an odd degree
    adj_matrix = mst.toarray() + mst.toarray().T
    degrees = np.count_nonzero(adj_matrix, axis=1)
    odd_degree_nodes_indices = np.where(degrees % 2 != 0)[0]

    # Step 4: Find a Minimum-Weight Perfect Matching on the odd-degree subgraph
    final_matching_pairs = find_minimum_weight_matching(dist_matrix, odd_degree_nodes_indices)

    # Step 5: Combine the MST and the matching edges to create a multigraph
    multigraph_adj = mst.toarray().astype(int) + mst.toarray().T.astype(int)
    for i, j in final_matching_pairs:
        multigraph_adj[i, j] += 1
        multigraph_adj[j, i] += 1

    # Step 6: Find an Eulerian Circuit in the multigraph
    print("\n--- Step 6 & 7: Finding Eulerian Circuit and Final Tour ---")
    start_node = 0
    # This now calls the REVISED, more robust function
    eulerian_circuit = find_eulerian_circuit(multigraph_adj, start_node)
    print(f"Eulerian Circuit (Revised): {eulerian_circuit}")

    # Step 7: Convert the Eulerian circuit to a Hamiltonian circuit (the TSP tour)
    path = []
    visited = set()
    for node in eulerian_circuit:
        if node not in visited:
            path.append(node)
            visited.add(node)
    path.append(path[0])

    tour_length = 0
    for i in range(len(path) - 1):
        tour_length += dist_matrix[path[i], path[i + 1]]

    print(f"\nFinal TSP Tour: {path}")
    print(f"Total Tour Length: {tour_length:.2f}")

    # Visualize the final tour
    visualize_final_tour(cities, path, tour_length)


def find_minimum_weight_matching(dist_matrix, odd_nodes):
    """Helper function for Step 4"""
    # ... (code for matching, unchanged)
    odd_node_dist_matrix = dist_matrix[odd_nodes][:, odd_nodes]
    unique_matchings = set()
    if len(odd_nodes) > 0:
        for p in itertools.permutations(range(len(odd_nodes))):
            pairs = tuple(sorted(tuple(sorted((p[i], p[i + 1]))) for i in range(0, len(p), 2)))
            unique_matchings.add(pairs)
    min_weight = float('inf')
    best_matching = None
    for matching in unique_matchings:
        weight = 0
        for p1, p2 in matching:
            weight += odd_node_dist_matrix[p1, p2]
        if weight < min_weight:
            min_weight = weight
            best_matching = matching
    final_pairs = []
    if best_matching:
        for p1, p2 in best_matching:
            final_pairs.append((odd_nodes[p1], odd_nodes[p2]))
    return final_pairs


def find_eulerian_circuit(multigraph_adj, start_node):
    """
    A more robust implementation of Hierholzer's algorithm.
    """
    # We work on a copy to be able to modify it
    graph = multigraph_adj.copy()
    stack = [start_node]
    path = []

    while stack:
        vertex = stack[-1]

        # Check if there's any edge left to explore from this vertex
        found_edge = False
        for i in range(len(graph[vertex])):
            if graph[vertex, i] > 0:
                # If an edge is found, add the neighbor to the stack and remove the edge
                stack.append(i)
                graph[vertex, i] -= 1
                graph[i, vertex] -= 1
                found_edge = True
                break  # Move to the new vertex

        if not found_edge:
            # If there are no more edges, we've explored all paths from this vertex.
            # Pop it from the stack and add it to our final circuit.
            path.append(stack.pop())

    return path[::-1]  # The path is constructed in reverse


def calculate_distance_matrix(cities):
    """Calculates the Euclidean distance matrix."""
    return squareform(pdist(cities, 'euclidean'))


def visualize_final_tour(cities, path, tour_length):
    """Plots the final TSP tour."""
    plt.figure(figsize=(8, 8))
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        plt.plot([cities[p1, 0], cities[p2, 0]], [cities[p1, 1], cities[p2, 1]], 'b-')
    plt.scatter(cities[:, 0], cities[:, 1], c='red', zorder=3)
    for i, city in enumerate(cities):
        plt.text(city[0] + 0.5, city[1] + 0.5, str(i), fontsize=12)
    plt.title(f"Final TSP Tour (Christofides)\nLength: {tour_length:.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()


# --- Main execution block ---
if __name__ == '__main__':
    sample_cities = np.array([
        [82, 76], [96, 44], [50, 5], [49, 8], [13, 7], [29, 89],
        [58, 30], [84, 39], [14, 24], [2, 39]
    ])
    solve_tsp_christofides(sample_cities)