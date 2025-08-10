' Neural Networks: CNN '

# import libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# load and preprocessing the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test/ 255.0 # normalize pixel value
x_train = x_train.reshape(-1, 28, 28, 1) # reshape for CNN
x_test = x_test.reshape(-1, 28, 28, 1)

# define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)),
    layers.MaxPooling2D((2, 2)), 
    layers.Conv2D(64, (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)), 
    layers.Conv2D(64, (3, 3), activation = 'relu'),
    layers.Flatten(), 
    layers.Dense(64, activation = 'relu'), 
    layers.Dense(10, activation = 'softmax') # 10 classes for digits 0-9
])

# compile the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# train the model
model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_split = 0.2)

# evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)