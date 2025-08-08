' Supervised Learning Algorithm: Polynomial Regression '

# import libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# load sample data ( experience vs salary )
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([45000, 50000, 60000, 80000, 150000, 200000, 300000, 400000, 500000, 600000])

# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# transform features into polynomial features
poly = PolynomialFeatures(degree= 2) #this means we will transform the feature X into a second degree polynomial
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# initialize and train the model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# make predictions
y_pred = model.predict(X_test_poly)

# evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)
print("Predicted values: ", y_pred)
