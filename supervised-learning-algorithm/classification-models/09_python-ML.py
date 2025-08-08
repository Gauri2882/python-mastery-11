' Ensemble Learning: Gradient Boosting '

# import libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# load data ( hours studied and prior grades vs. pass/fail)
X = np.array([[1, 50], [2, 60], [3, 55], [4, 65], [5, 70], [6, 75], [7, 80], [8, 85], [9, 89], [10, 92]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]) # 0 = fail, 1 = pass

# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# train the model 
model = GradientBoostingClassifier(n_estimators= 100, learning_rate= 0.1, random_state= 42) # n_estimators means number of decision trees here we used 100, and random state reproducility, learning rate is controls contribution of each tree for final predictions
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)