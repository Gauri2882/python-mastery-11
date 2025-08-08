' Unsupervised clustering: Gaussian Mixture Models (GMM) '

# import libraries
from sklearn.mixture import GaussianMixture
import numpy as np

# sample data ( points in 2D spaces )
X = np.array([
    [1, 2], [2, 2], [2, 3], 
    [8, 7], [8, 8], [25, 80]
])

# initialize and fit the model
gmm = GaussianMixture(n_components= 2, random_state= 42)
gmm.fit(X)

# get the cluster labels and probabilities
labels = gmm.predict(X)
probs = gmm.predict_proba(X)

print("Cluster Labels:", labels)
print("Cluster Probabilties:\n", probs)