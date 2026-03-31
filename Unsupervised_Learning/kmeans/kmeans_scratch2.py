import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

import numpy as np

from sklearn.metrics import silhouette_score


#generate dataset

X, y = make_blobs(
    n_samples=300,    # Number of points
    centers=20,        # Number of blobs
    cluster_std=0.70, # Standard deviation (controls spread)
    random_state=0    # Ensures reproducible results
)

# 2. Visualize generated data (unlabeled)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Synthetic Blob Data")
plt.show()

wss = []
silhotte= []
k_values = range(2,23)
for i in k_values:

          kmeans = KMeans(n_clusters=i, n_init=10) # 4 centers
          kmeans.fit(X)
          y_kmeans = kmeans.predict(X)
          wss.append(round(kmeans.inertia_,3))
          silhotte.append(silhouette_score(X,y_kmeans))


plt.figure(figsize=(8, 5))
plt.plot(k_values,wss)
plt.ylabel('inertia')
plt.xlabel('kvalues')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(k_values,silhotte)
plt.ylabel('silhoutte')
plt.xlabel('kvalues')
plt.show()



kmeans = KMeans(n_clusters=17 , n_init=10) # 4 centers
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


# 4. Plot Results with Centroids
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title("K-Means Clustering Result")
plt.show()
print(wss)