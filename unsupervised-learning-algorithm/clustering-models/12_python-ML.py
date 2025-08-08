' Unsupervised Learning: Hierarichal Clustering '

# import libraries
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

# sample data ( points in 2D spaces )
X = np.array([
    [1, 2], [1, 4], [1, 0], 
    [10, 2], [10, 4], [10, 0]
])

# perform hierarichal/agglomerative clustering
Z = linkage(X, method= 'ward') # ward minimizes variance with clusters

# plot dendograms
plt.figure(figsize= (8, 4))
dendrogram(Z)
plt.title("Dendograms for Hierarichal Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distances")
plt.show()