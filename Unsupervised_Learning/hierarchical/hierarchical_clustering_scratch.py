import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

import numpy as np

from sklearn.metrics import silhouette_score


#generate dataset

X, y = make_blobs(
    n_samples=300,    # Number of points
    centers=4,        # Number of blobs
    cluster_std=0.70, # Standard deviation (controls spread)
    random_state=0    # Ensures reproducible results
)

# 2. Visualize generated data (unlabeled)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Synthetic Blob Data")
plt.show()

def euclidean_distance(a,b):          
         return np.linalg.norm(a - b)


def manhattan_distance(a,b):
        return np.sum(np.abs(a-b))

euclidean =[]
min_distance = float('inf')
closest_pair = []
cluster =[]
for i in range(len(X)):
        for j in range(len(X)):
                if j < i :
                    dist = round(euclidean_distance(X[i],X[j]),4)
                    euclidean.append((i,j,dist))
                    if min_distance > dist :
                           min_distance = dist
                           closest_pair = (i,j)

cluster.append(closest_pair)



print(euclidean, min_distance, closest_pair,cluster)