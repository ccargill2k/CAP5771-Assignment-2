# Import libraries
import numpy as np
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

# Load in the data
seeds = np.loadtxt('seeds.txt')
if seeds.shape[1] == 8:
    seeds = seeds[:, :-1]

# Random K selection
def init_cents(seeds, k):
    indices = np.random.choice(seeds.shape[0], k, replace=False)
    return seeds[indices].copy()

# Calc Euclidean Distance 
def assign_clusters(seeds, centroids):
    distances = np.zeros((seeds.shape[0], len(centroids)))
    for i, centroid in enumerate(centroids):
        for j, point in enumerate(seeds):
            distances[j, i] = euclidean(point, centroid)
    return np.argmin(distances, axis=1)

def recalc_cents(seeds, labels, k):
    centroids = np.zeros((k, seeds.shape[1]))
    for i in range(k):
        cluster_points = seeds[labels == i]
        if len(cluster_points) > 0:
            centroids[i] = cluster_points.mean(axis=0)
        else:
            centroids[i] = seeds[np.random.choice(seeds.shape[0])]
    return centroids

# SSE
def calc_sse(seeds, labels, centroids):
    sse = 0
    for i, centroid in enumerate(centroids):
        cluster_points = seeds[labels == i]
        for point in cluster_points:
            sse += euclidean(point, centroid) ** 2
    return sse

# K Means
def kmeans(seeds, k, max_iterations=100, tolerance=0.001):
    centroids = init_cents(seeds, k)
    prev_sse = float('inf')
    
    for iteration in range(max_iterations):
        labels = assign_clusters(seeds, centroids)
        centroids = recalc_cents(seeds, labels, k)
        sse = calc_sse(seeds, labels, centroids)
        
        if abs(prev_sse - sse) < tolerance:
            return centroids, labels, sse, iteration + 1
        prev_sse = sse
    
    return centroids, labels, sse, max_iterations

# Loop
k_values = [3, 5, 7]
n_runs = 10

# Print Results
results = {}
for k in k_values:
    sse_values = []
    for run in range(n_runs):
        centroids, labels, sse, iterations = kmeans(seeds, k)
        sse_values.append(sse)
    avg_sse = np.mean(sse_values)
    results[k] = avg_sse
for k in k_values:
    print(f"k={k}: Average SSE = {results[k]:.4f}")