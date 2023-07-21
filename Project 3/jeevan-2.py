#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

def calculate_sse(X: np.ndarray, labels: np.ndarray, centers: np.ndarray):
    "Calculates SSE error based on the given inputs"
    sse = 0
    for i in range(len(X)):
        dist = np.linalg.norm(X[i]-centers[labels[i]])
        sse += dist**2
    return sse

def run_kmeans(dataset_path: str, i_max: int=20):
    "Performs the kmeans clustering in the given criterian and gives desired output"
    X = np.loadtxt(dataset_path)[:, :-1]  # Excluding the last column from features
    sse_values = list() # To store SSE values for all the K values (in given range)
    for k in range(2, 11):
        sse_sum, i = 0, 0
        while i < i_max:
            kmeans = KMeans(n_clusters=k, init='random', n_init=1)
            kmeans.fit(X)
            labels = kmeans.predict(X)
            centers = kmeans.cluster_centers_ 
            sse = calculate_sse(X, labels, centers)
            sse_sum += sse
            i += 1
        sse_avg = sse_sum / i_max
        sse_values.append(sse_avg)
        print(f"For k = {k} After 20 iterations: SSE error = {sse_avg:.4f}")
    
    # Plot SSE vs K values (chart)
    plt.plot(range(2, 11), sse_values, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('SSE Error')
    plt.title('SSE vs K chart')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python kmeans_clustering.py <data_file>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    run_kmeans(data_file)
