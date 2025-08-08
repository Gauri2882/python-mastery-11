' Unsupervised Dimesionality Reduction: Principal Component Algorithm (PCA) '

# import libraries
from sklearn.decomposition import PCA
import numpy as np

# sample data (e.g., points in 3D space)
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7], [5, 7, 8]])

# initialize and fit the model
pca = PCA(n_components= 2) # reducing to 2 dimensions
X_reduced = pca.fit_transform(X)

print("Reduced Data:\n", X_reduced)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)