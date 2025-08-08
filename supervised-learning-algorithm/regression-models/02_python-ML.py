' Supervised algorithm: Ridge and Lasso regression '

# importing necessary libraries
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np 

# sample data
X = np.array([[1400], [1600], [1700], [1875], [1100], [1550], [2350], [2450], [1425], [1700]])  # house size in sqft
y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 240000, 409000])  # house prices in dollar

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# Ridge regression
ridge_model = Ridge(alpha= 1.0) # alpha controls the L2 regularization strength
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)
print("Ridge Mean Squared Error: ", ridge_mse)

# lasso regression
lasso_model = Lasso(alpha= 0.1) # alpha controls the L2 regularization strength
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)
print("Lassp Mean Squared Error: ", lasso_mse)
