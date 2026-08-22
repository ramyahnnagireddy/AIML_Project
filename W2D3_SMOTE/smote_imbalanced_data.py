"""
W2D3: Handling Imbalanced Data with SMOTE

This script demonstrates:
1. Creating an imbalanced classification dataset.
2. Splitting data using stratification.
3. Applying SMOTE only to the training data.
4. Checking class distribution before and after SMOTE.
5. Training a Logistic Regression model.
6. Evaluating the model on untouched test data.
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20


def create_dataset():
    """Create an imbalanced binary classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        weights=[0.90, 0.10],
        random_state=RANDOM_STATE,
    )

    return X, y


def show_class_distribution(y, title):
    """Print the class distribution."""
    print(f"\n===== {title} =====")

    unique, counts = np.unique(y, return_counts=True)

    for label, count in zip(unique, counts):
        print(f"Class {label}: {count}")


def split_data(X, y):
    """Split data while preserving class proportions."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def apply_smote(X_train, y_train):
    """Balance only the training data using SMOTE."""
    smote = SMOTE(random_state=RANDOM_STATE)

    return smote.fit_resample(X_train, y_train)


def train_model(X_train, y_train):
    """Train Logistic Regression on the balanced training data."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on untouched test data."""
    y_pred = model.predict(X_test)

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def main():
    """Run the complete SMOTE workflow."""

    # Create imbalanced dataset
    X, y = create_dataset()

    print("===== ORIGINAL DATASET =====")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    show_class_distribution(y, "ORIGINAL CLASS DISTRIBUTION")

    # Split data before applying SMOTE
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n===== BEFORE SMOTE =====")
    print(f"Training samples: {len(y_train)}")
    print(f"Testing samples: {len(y_test)}")

    show_class_distribution(
        y_train,
        "TRAINING CLASS DISTRIBUTION BEFORE SMOTE",
    )

    # Apply SMOTE only to training data
    X_train_smote, y_train_smote = apply_smote(
        X_train,
        y_train,
    )

    print("\n===== AFTER SMOTE =====")
    print(f"Training samples after SMOTE: {len(y_train_smote)}")

    show_class_distribution(
        y_train_smote,
        "TRAINING CLASS DISTRIBUTION AFTER SMOTE",
    )

    # Train model
    model = train_model(
        X_train_smote,
        y_train_smote,
    )

    # Evaluate on untouched test data
    evaluate_model(
        model,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()