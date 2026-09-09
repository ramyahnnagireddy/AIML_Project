"""
W3D3: Decision Trees & Random Forests

This script:
1. Loads a real classification dataset.
2. Splits the data into training and testing sets.
3. Trains a Decision Tree using Gini impurity.
4. Evaluates the Decision Tree.
5. Visualizes the Decision Tree.
6. Tunes max_depth to reduce overfitting.
7. Trains a Random Forest.
8. Compares the models using evaluation metrics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# -----------------------------
# Constants
# -----------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# 1. Load the dataset
# -----------------------------

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("Dataset shape:", X.shape)
print("Target classes:", data.target_names)


# -----------------------------
# 2. Train-test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# -----------------------------
# 3. Train Decision Tree
# -----------------------------
# criterion="gini" uses Gini impurity to decide the best split.

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    random_state=RANDOM_STATE,
)

decision_tree.fit(X_train, y_train)

dt_train_predictions = decision_tree.predict(X_train)
dt_test_predictions = decision_tree.predict(X_test)


# -----------------------------
# 4. Evaluate Decision Tree
# -----------------------------

dt_train_accuracy = accuracy_score(y_train, dt_train_predictions)
dt_test_accuracy = accuracy_score(y_test, dt_test_predictions)

print("\n--- Decision Tree ---")
print("Training Accuracy:", round(dt_train_accuracy, 4))
print("Testing Accuracy:", round(dt_test_accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, dt_test_predictions))


# -----------------------------
# 5. Visualize Decision Tree
# -----------------------------

plt.figure(figsize=(20, 10))

plot_tree(
    decision_tree,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    max_depth=3,
    fontsize=7,
)

plt.title("Decision Tree - First 3 Levels")
plt.tight_layout()

tree_path = OUTPUT_DIR / "decision_tree.png"
plt.savefig(tree_path, dpi=150)
plt.close()

print("Decision Tree visualization saved to:", tree_path)


# -----------------------------
# 6. Tune Decision Tree
# -----------------------------
# Limiting max_depth helps control overfitting.

tuned_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_split=5,
    random_state=RANDOM_STATE,
)

tuned_tree.fit(X_train, y_train)

tuned_train_predictions = tuned_tree.predict(X_train)
tuned_test_predictions = tuned_tree.predict(X_test)

tuned_train_accuracy = accuracy_score(
    y_train,
    tuned_train_predictions,
)

tuned_test_accuracy = accuracy_score(
    y_test,
    tuned_test_predictions,
)

print("\n--- Tuned Decision Tree ---")
print("Training Accuracy:", round(tuned_train_accuracy, 4))
print("Testing Accuracy:", round(tuned_test_accuracy, 4))


# -----------------------------
# 7. Train Random Forest
# -----------------------------

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
)

random_forest.fit(X_train, y_train)

rf_train_predictions = random_forest.predict(X_train)
rf_test_predictions = random_forest.predict(X_test)

rf_train_accuracy = accuracy_score(
    y_train,
    rf_train_predictions,
)

rf_test_accuracy = accuracy_score(
    y_test,
    rf_test_predictions,
)


# -----------------------------
# 8. Evaluate Random Forest
# -----------------------------

print("\n--- Random Forest ---")
print("Training Accuracy:", round(rf_train_accuracy, 4))
print("Testing Accuracy:", round(rf_test_accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, rf_test_predictions))


# -----------------------------
# 9. Compare Models
# -----------------------------

results = pd.DataFrame(
    {
        "Model": [
            "Decision Tree",
            "Tuned Decision Tree",
            "Random Forest",
        ],
        "Train Accuracy": [
            dt_train_accuracy,
            tuned_train_accuracy,
            rf_train_accuracy,
        ],
        "Test Accuracy": [
            dt_test_accuracy,
            tuned_test_accuracy,
            rf_test_accuracy,
        ],
        "Precision": [
            precision_score(y_test, dt_test_predictions),
            precision_score(y_test, tuned_test_predictions),
            precision_score(y_test, rf_test_predictions),
        ],
        "Recall": [
            recall_score(y_test, dt_test_predictions),
            recall_score(y_test, tuned_test_predictions),
            recall_score(y_test, rf_test_predictions),
        ],
        "F1 Score": [
            f1_score(y_test, dt_test_predictions),
            f1_score(y_test, tuned_test_predictions),
            f1_score(y_test, rf_test_predictions),
        ],
    }
)

print("\n--- Model Comparison ---")
print(results.to_string(index=False))

results_path = OUTPUT_DIR / "model_comparison.csv"
results.to_csv(results_path, index=False)

print("\nModel comparison saved to:", results_path)


# -----------------------------
# 10. Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, rf_test_predictions)

print("\nRandom Forest Confusion Matrix:")
print(cm)
