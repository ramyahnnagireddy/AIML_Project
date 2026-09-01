"""
W3D2: Logistic Regression & Classification

This script demonstrates:
1. Loading a classification dataset
2. Train/test splitting
3. Logistic Regression training
4. Class prediction
5. Probability prediction using sigmoid-based output
6. Model evaluation
7. Multi-class classification
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------
# 1. Load the Iris dataset
# ---------------------------------------------------------
iris = load_iris()

X = iris.data
y = iris.target

print("===== DATASET INFORMATION =====")
print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Class names: {iris.target_names}")
print()


# ---------------------------------------------------------
# 2. Split the dataset into training and testing sets
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("===== TRAIN/TEST SPLIT =====")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print()


# ---------------------------------------------------------
# 3. Create and train Logistic Regression model
# ---------------------------------------------------------
model = LogisticRegression(
    max_iter=200,
    random_state=42,
)

model.fit(X_train, y_train)

print("===== MODEL TRAINING =====")
print("Logistic Regression model trained successfully.")
print()


# ---------------------------------------------------------
# 4. Make class predictions
# ---------------------------------------------------------
y_pred = model.predict(X_test)

print("===== PREDICTIONS =====")
print("First 10 actual classes:   ", y_test[:10])
print("First 10 predicted classes:", y_pred[:10])
print()


# ---------------------------------------------------------
# 5. Predict class probabilities
# ---------------------------------------------------------
y_probability = model.predict_proba(X_test)

print("===== PREDICTED PROBABILITIES =====")
print(y_probability[:5])
print()


# ---------------------------------------------------------
# 6. Evaluate the model using accuracy
# ---------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("===== MODEL EVALUATION =====")
print(f"Accuracy: {accuracy:.4f}")
print()


# ---------------------------------------------------------
# 7. Display confusion matrix
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

print("===== CONFUSION MATRIX =====")
print(cm)
print()


# ---------------------------------------------------------
# 8. Display classification report
# ---------------------------------------------------------
print("===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
    )
)


# ---------------------------------------------------------
# 9. Display model coefficients
# ---------------------------------------------------------
print("===== MODEL COEFFICIENTS =====")
print(model.coef_)
print()

print("===== MODEL INTERCEPT =====")
print(model.intercept_)
