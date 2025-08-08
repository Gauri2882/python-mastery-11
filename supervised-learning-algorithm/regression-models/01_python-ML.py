' Supervised algorithm: Linear regression '

# importing necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # splitting data
from sklearn.metrics import mean_squared_error # lower value of this indicate better performance
import numpy as np 

# sample data (house size vs house price)
X = np.array([[1400], [1600], [1700], [1875], [1100], [1550], [2350], [2450], [1425], [1700]])  # house size in sqft
y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 240000, 409000])

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# model training
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}\nPredicted Values: {y_pred}")