import numpy as np
from sklearn.datasets import make_blobs

# Generate dataset
X, _ = make_blobs(n_samples=50, centers=4, cluster_std=0.70, random_state=0)

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Step 1: Initialize clusters (each point is its own cluster)
clusters = [[i] for i in range(len(X))]

# Step 2: Hierarchical clustering loop
while len(clusters) > 1:

    min_distance = float('inf')
    closest_clusters = (None, None)

    # Step 3: Find closest pair of clusters
    for i in range(len(clusters)):
        for j in range(i):
            cluster1 = clusters[i]
            cluster2 = clusters[j]

            # Single linkage: min distance between any points
            cluster_distance = float('inf')

            for p1 in cluster1:
                for p2 in cluster2:
                    dist = euclidean_distance(X[p1], X[p2])
                    if dist < cluster_distance:
                        cluster_distance = dist

            # Update global minimum
            if cluster_distance < min_distance:
                min_distance = cluster_distance
                closest_clusters = (i, j)

    # Step 4: Merge closest clusters
    i, j = closest_clusters

    clusters[min(i, j)].extend(clusters[max(i, j)])
    clusters.pop(max(i, j))

    # Optional: print progress
    print(f"Merged clusters {i} and {j}, remaining clusters: {len(clusters)}")

print("\nFinal cluster:", clusters)
