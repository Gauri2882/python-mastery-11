' Semi supervised Learning: Self Training'

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Generate synthetic dataset
X, y = make_classification(n_samples=200, n_features=5, random_state=42)
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, test_size=0.7, random_state=42)

# Train the model with labeled data
model = RandomForestClassifier(random_state=42)
model.fit(X_labeled, y_labeled)

# Perform self-training on unlabeled data
for _ in range(5):
    if len(X_unlabeled) == 0:
        print("No more unlabeled samples left.")
        break

    # Predict the probabilities on the unlabeled data
    probs = model.predict_proba(X_unlabeled)
    high_confidence_idx = np.where(np.max(probs, axis=1) > 0.9)[0]

    # If no high-confidence predictions, stop early
    if len(high_confidence_idx) == 0:
        print("No high-confidence predictions found. Stopping.")
        break

    # Add high-confidence predictions to labeled data
    X_labeled = np.vstack([X_labeled, X_unlabeled[high_confidence_idx]])
    y_labeled = np.hstack([y_labeled, model.predict(X_unlabeled[high_confidence_idx])])

    # Remove confident samples from the unlabeled dataset
    X_unlabeled = np.delete(X_unlabeled, high_confidence_idx, axis=0)

    # Re-train the model on the expanded labeled dataset
    model.fit(X_labeled, y_labeled)

# Final evaluation on a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)