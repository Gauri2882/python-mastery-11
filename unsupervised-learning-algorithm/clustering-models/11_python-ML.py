' Unsupervised Learning: K-Means Clustering'

# import libraries
from sklearn.cluster import KMeans
import numpy as np

# sample data ( points in 2D spaces )
X = np.array([
    [1, 2], [1, 4], [1, 0], 
    [10, 2], [10, 4], [10, 0]
])

# fit the model
kmeans = KMeans(n_clusters= 2, random_state= 42)
kmeans.fit(X)

# get the cluster centers and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print("Cluster Centers:\n", centroids)
print("Labels:", labels)