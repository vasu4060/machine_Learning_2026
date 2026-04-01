import numpy as np
from sklearn.datasets import make_blobs

# Generate dataset
X, _ = make_blobs(n_samples=100, centers=4, cluster_std=0.70, random_state=0)

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Parameters
eps = 0.5
min_samples = 5

# Initialize
n = len(X)
labels = np.full(n, -1)     
visited = np.zeros(n, dtype=bool)
cluster_id = 0

# Helper: find neighbors
def get_neighbors(point_idx):
    neighbors = []
    for j in range(n):
        if euclidean_distance(X[point_idx], X[j]) <= eps:
            neighbors.append(j)
    return neighbors

# Main loop
for i in range(n):
    if visited[i]:
        continue

    visited[i] = True
    neighbors = get_neighbors(i)

    # Not a core point → noise
    if len(neighbors) < min_samples:
        labels[i] = -1

    else:
        # Start new cluster
        cluster_id += 1
        labels[i] = cluster_id

        # Expand cluster (BFS style)
        queue = neighbors.copy()

        k = 0
        while k < len(queue):
            point = queue[k]

            if not visited[point]:
                visited[point] = True
                new_neighbors = get_neighbors(point)

                if len(new_neighbors) >= min_samples:
                    queue.extend(new_neighbors)

            # Assign cluster label
            if labels[point] == -1:
                labels[point] = cluster_id

            k += 1

print("Cluster labels:", labels)