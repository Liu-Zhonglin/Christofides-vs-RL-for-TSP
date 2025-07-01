import numpy as np


def solve_tsp_nearest_insertion(cities):
    """
    Solves the TSP using the Nearest Insertion heuristic.
    Takes city coordinates as input and returns the final tour and its length.
    """
    num_cities = len(cities)
    if num_cities < 2:
        return [], 0

    # Create the distance matrix from coordinates
    dist_matrix = np.sqrt(((cities[:, np.newaxis, :] - cities[np.newaxis, :, :]) ** 2).sum(axis=2))

    # Start with a tour of the first two cities
    tour = [0, 1, 0]
    used = {0, 1}

    # Iteratively insert the remaining cities
    while len(used) < num_cities:
        min_insert_cost = float('inf')
        best_node_to_insert = -1
        best_position = -1

        # Find the unused node k that is closest to any node j already in the tour
        min_dist_to_tour = float('inf')
        closest_node_k = -1

        for k in range(num_cities):
            if k not in used:
                for j in tour[:-1]:  # For each node already in the tour
                    if dist_matrix[k, j] < min_dist_to_tour:
                        min_dist_to_tour = dist_matrix[k, j]
                        closest_node_k = k

        node_to_insert = closest_node_k

        # Find the best position to insert this node
        for i in range(len(tour) - 1):
            u = tour[i]
            v = tour[i + 1]
            cost = dist_matrix[u, node_to_insert] + dist_matrix[node_to_insert, v] - dist_matrix[u, v]
            if cost < min_insert_cost:
                min_insert_cost = cost
                best_position = i + 1

        tour.insert(best_position, node_to_insert)
        used.add(node_to_insert)

    # Calculate final tour length
    tour_length = sum(dist_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))

    return tour, tour_length