import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

def coordinate_generator():
    """
    This function generates the coordinates in random as input.

    Parameters:
        None

    :return: coordinates
    """
    # Coordinates for Boston
    min_latitude = 42.2279
    max_latitude = 42.4061
    min_longitude = -71.1912
    max_longitude = -70.9865
    num_clusters = 7

    #Converting the min and max latitude to an area with length and height
    min_length = 0
    min_width = 0
    max_length = (max_latitude - min_latitude) * 69  #As one degree is 69 miles
    max_height = (max_longitude - min_longitude) * 69

    num_points = 60
    np.random.seed(16)
    latitudes = np.round(np.random.uniform(min_length, max_length, num_points), 6)
    longitudes = np.round(np.random.uniform(min_width, max_height, num_points), 6)
    coordinates = np.column_stack((latitudes,longitudes))
    K_Means(coordinates,num_clusters,longitudes,latitudes)

def K_Means(coordinates,num_clusters,longitudes,latitudes):
    """
    This function takes the input parameters from the previous code and clusters all the coordinates which are together

    :param coordinates : Random generated coordinates.
    :return: value,cluster_number
    """
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_assignment = kmeans.fit_predict(coordinates)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    cluster_counts = {i: 0 for i in range(num_clusters)}

    modified_latitudes = []
    modified_longitudes = []

    # Loop through the data points and assign them to clusters with a maximum of 10 points
    for i, (lat, lon) in enumerate(zip(latitudes, longitudes)):
        cluster_id = cluster_labels[i]
        if cluster_counts[cluster_id] < 10:
            modified_latitudes.append(lat)
            modified_longitudes.append(lon)
            cluster_counts[cluster_id] += 1

    # Convert the modified latitudes and longitudes back to arrays
    modified_coordinates = np.column_stack((modified_latitudes, modified_longitudes))

    # Initialize a new KMeans model for the modified data
    kmeans_modified = KMeans(n_clusters=num_clusters)

    # Fit the modified model to the data
    kmeans_modified.fit(modified_coordinates)

    # Get the modified cluster labels and centers
    cluster_labels_modified = kmeans_modified.labels_
    cluster_centers_modified = kmeans_modified.cluster_centers_

    cluster_coordinates = {i: [] for i in range(num_clusters)}

    # Populate the dictionary with coordinates
    for i, (lat, lon) in enumerate(zip(modified_latitudes, modified_longitudes)):
        cluster_id = cluster_labels[i]
        if len(cluster_coordinates[cluster_id]) < 10:
            cluster_coordinates[cluster_id].append((lat, lon))

    cluster_number = 0
    for key, val in cluster_coordinates.items():
        cluster_number = cluster_number + 1
        Distance = matrix_distance(val, cluster_number)

def matrix_distance(value,cluster_number):
    """
    To create a symmetric matrix that calculates the distance between each nodes in a cluster.
    :param value: A cluster with nodes
    :param cluster_number: Which cluster number we are referring
    :return: dist_matrix, cluster_number
    """
    dist_matrix = np.zeros((len(value), len(value)), dtype=int)

    for i in range(len(value)):
        for j in range(i+1,len(value)):
            x1,y1 = value[i]
            x2,y2 = value[j]
            distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)
            dist_matrix[i][j] = distance
            dist_matrix[j][i] = distance
    close_neighbours = nearest_neighbor_tsp(dist_matrix,cluster_number)

def nearest_neighbor_tsp(distance_matrix,cluster_number):
    """
    This function is used to find the nearest neighbor for a cluster
    :param distance_matrix: Symmetric matrix containing the distance between each points
    :param cluster_number: Cluster number
    :return: None
    """
    num_nodes = len(distance_matrix)
    unvisited_nodes = set(range(1, num_nodes))  # Excluding the starting node (index 0)
    tour = [0]  # Start the tour from node 0
    current_node = 0
    total_distance = 0

    while unvisited_nodes:
        nearest_neighbor = min(unvisited_nodes, key=lambda node: distance_matrix[current_node][node])
        total_distance += distance_matrix[current_node][nearest_neighbor]
        tour.append(nearest_neighbor)
        unvisited_nodes.remove(nearest_neighbor)
        current_node = nearest_neighbor

    # Return to the starting node to complete the tour
    tour.append(0)
    print("For the cluster "+str(cluster_number)+ " the optimal route is "+str(tour)+"and the total distance covered is "+str(total_distance)+" miles")

if __name__ == "__main__":
    Cluter_cordinates = coordinate_generator()
