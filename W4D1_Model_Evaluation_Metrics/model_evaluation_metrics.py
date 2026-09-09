"""
W4D1: Model Evaluation Metrics
Precision, Recall, F1-Score, ROC-AUC,
Stratified K-Fold Cross-Validation and Learning Curve.
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Reproducibility
RANDOM_STATE = 42


def load_data():
    """Load the Breast Cancer dataset."""
    data = load_breast_cancer()
    return data.data, data.target


def train_model(X_train, y_train):
    """Train a Logistic Regression model."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    )
    model.fit(X_train, y_train)
    return model


def calculate_metrics(model, X_test, y_test):
    """Calculate common classification evaluation metrics."""
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_probability),
    }

    return metrics, confusion_matrix(y_test, y_pred)


def perform_stratified_cv(X, y):
    """Evaluate the model using Stratified K-Fold cross-validation."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    accuracy_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    precision_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="precision",
    )

    recall_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="recall",
    )

    roc_auc_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
    )

    return {
        "Accuracy": accuracy_scores,
        "Precision": precision_scores,
        "Recall": recall_scores,
        "ROC-AUC": roc_auc_scores,
    }


def plot_learning_curve(X, y):
    """Plot training and validation scores to diagnose model fit."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_sizes, train_scores, validation_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
    plt.plot(
        train_sizes,
        validation_mean,
        marker="o",
        label="Validation Accuracy",
    )

    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - Logistic Regression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("learning_curve.png", dpi=150)
    plt.show()

    return train_mean, validation_mean


def main():
    """Run the complete model evaluation workflow."""

    # Load dataset
    X, y = load_data()

    print("=" * 60)
    print("W4D1: MODEL EVALUATION METRICS")
    print("=" * 60)

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\n--- Train/Test Split ---")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train model
    model = train_model(X_train, y_train)

    # Calculate evaluation metrics
    metrics, matrix = calculate_metrics(model, X_test, y_test)

    print("\n--- Test Set Metrics ---")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    # Stratified K-Fold Cross-Validation
    cv_results = perform_stratified_cv(X, y)

    print("\n--- Stratified 5-Fold Cross-Validation ---")

    for metric_name, scores in cv_results.items():
        print(f"{metric_name}:")
        print(f"  Scores: {np.round(scores, 4)}")
        print(f"  Mean: {scores.mean():.4f}")

    # Learning curve
    print("\n--- Learning Curve ---")
    train_scores, validation_scores = plot_learning_curve(X, y)

    print(f"Final training accuracy: {train_scores[-1]:.4f}")
    print(f"Final validation accuracy: {validation_scores[-1]:.4f}")

    print("\nLearning curve saved as: learning_curve.png")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()

    