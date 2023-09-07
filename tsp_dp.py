#import itertools
import math
import time


def euclidean_distance(point1, point2):
    lat1, lon1 = point1
    lat2, lon2 = point2
    return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

def tsp_dp(university, cluster):
    start_time = time.time()
    cluster.append(university)
    n = len(cluster)

    # Initialize the DP table with -1 (unvisited)
    dp_table = [[-1] * n for _ in range(1 << n)]
    best_order = [[-1] * n for _ in range(1 << n)]

    # Initialize the base case: distance from university to all customers
    for i in range(n):
        dp_table[1 << i][i] = euclidean_distance(university, cluster[i])

    # Iterate through all subproblems
    for mask in range(1, (1 << n)):
        for i in range(n):
            if mask & (1 << i) != 0:
                for j in range(n):
                    if i != j and dp_table[mask][i] != -1:
                        new_mask = mask | (1 << j)
                        if dp_table[new_mask][j] == -1 or dp_table[new_mask][j] > dp_table[mask][i] + euclidean_distance(cluster[i], cluster[j]):
                            dp_table[new_mask][j] = dp_table[mask][i] + euclidean_distance(cluster[i], cluster[j])
                            best_order[new_mask][j] = i

    # Find the optimal path using the DP table and best_order
    mask = (1 << n) - 1
    last_node = -1
    for i in range(n):
        if best_order[mask][i] != -1:
            last_node = i
            break

    path = [university]
    while last_node != -1:
        path.append(cluster[last_node])
        new_mask = mask ^ (1 << last_node)
        last_node = best_order[mask][last_node]
        mask = new_mask

    # Calculate the total distance for the efficient path
    total_distance = 0
    for i in range(1, len(path)):
        total_distance += euclidean_distance(path[i-1], path[i])

    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken:", time_taken, "seconds")

    return path, total_distance

# Sample input
university = (42.3398, -71.0903)  # University coordinates (latitude, longitude)
cluster = [
    (42.310632, -71.107837),
    (42.328841, -71.097946),
    (42.312014, -71.109233),
    (42.349971, -71.118186),
    (42.330901, -71.136517),
    (42.401119, -71.104698)
]

efficient_path, total_distance = tsp_dp(university, cluster)
print("Efficient Path using DP:", efficient_path)
print("Total Distance:", total_distance)
