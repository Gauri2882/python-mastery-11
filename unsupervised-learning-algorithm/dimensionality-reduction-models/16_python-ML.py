'Unsupervised Dimensionality Reduction Algorithm: t-Distributed Stochastic Neighbor Embedding (t-SNE)'

# import libraries
from sklearn.manifold import TSNE
import numpy as np

# sample data (e.g., points in high-dimensional space)
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7], [5, 7, 8], [8, 9, 10]])

# fit the model
tsne = TSNE(n_components= 2, perplexity= 5, random_state= 42)
X_reduced = tsne.fit_transform(X)

print("Reduced Data:\n", X_reduced)