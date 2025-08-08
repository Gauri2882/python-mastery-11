' Unsupervised Dimensionality Reduction: Autoencoders '

# import libraries
import numpy as np
from tensorflow.keras import layers, models

# sample data (points in 5D space)
X = np.array([
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [5, 6, 7, 8, 9],
    [5, 7, 8, 9, 10],
    [8, 9, 10, 11, 12]
])

# define dimensions
input_dim = X.shape[1]
encoding_dim = 2  # compress to 2 dimensions

# encoder
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)

# decoder
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

# autoencoder model
autoencoder = models.Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# train
autoencoder.fit(X, X, epochs=100, batch_size = 2, verbose=0)

# encoder model to extract reduced dimensions
encoder = models.Model(inputs=input_layer, outputs=encoded)

# transform data
X_reduced = encoder.predict(X)
print("Reduced dimensions:\n", X_reduced)
