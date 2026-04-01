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
#plt.show()

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


clusters = [[i] for i in range(len(X))]

clusters[min(closest_pair[0], closest_pair[1])].extend(clusters[max(closest_pair[0], closest_pair[1])])
clusters.pop(max(closest_pair[0], closest_pair[1])) 

min_distance2 = float('inf')

closest_cluster = []
for cluster1 in clusters:
       for cluster2 in clusters:
              if cluster1 > cluster2:
                     for point1 in cluster1:
                            for point2 in cluster2:
                                   
                                        dist = round(euclidean_distance(X[point1],X[point2]),4)
                                        if dist < min_distance2 :
                                            min_distance2 = dist
                                           
                                            closest_cluster = (cluster1,cluster2)

index1 = clusters.index(cluster1)
index2 = clusters.index(cluster2)
clusters[index1].extend(clusters[index2])
clusters.pop(index2)
