# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 02:16:48 2023

@author: ivarr
"""

import itertools
import math
import time

#def calculate_distance(loc1, loc2):
#    return dist(loc1, loc2)

def euclidean_distance(point1, point2):
    lat1, lon1 = point1
    lat2, lon2 = point2
    return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

def find_optimal_path(university, cluster):
    start_time = time.time()
    cluster.append(university)
    min_distance = float('inf')
    optimal_path = []

    # Generate all permutations of customer locations
    for path_permutation in itertools.permutations(cluster):
        total_distance = 0

        # Calculate the total distance for the current permutation
        for i in range(len(path_permutation) - 1):
            total_distance += euclidean_distance(path_permutation[i], path_permutation[i + 1])

        # Add the distance from the last customer location back to the university
        total_distance += euclidean_distance(path_permutation[-1], university)

        # Update the optimal path if the current permutation yields a shorter distance
        if total_distance < min_distance:
            min_distance = total_distance
            optimal_path = path_permutation
    
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken:", time_taken, "seconds")

    return optimal_path

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

optimal_path = find_optimal_path(university, cluster)
print("Optimal Path using Brute force method:", optimal_path)


total_distance = 0
for i in range(1, len(optimal_path)):
    total_distance += euclidean_distance(optimal_path[i-1], optimal_path[i])

print("Total Distance:", total_distance)