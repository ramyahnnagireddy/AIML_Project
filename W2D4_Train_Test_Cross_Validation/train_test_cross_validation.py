"""
W2D4: Train/Test Split & Cross-Validation

This script demonstrates:
1. Train/test splitting
2. Stratified train/test splitting
3. K-Fold cross-validation
4. Stratified K-Fold cross-validation
5. Model evaluation using cross-validation
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import accuracy_score
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

def main():
    """Run train/test split and cross-validation experiments."""

    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    print("===== DATASET =====")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # ---------------------------------------------------------
    # 1. Train/Test Split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
       X,
       y,
       test_size=TEST_SIZE,
       random_state=RANDOM_STATE,
       shuffle=True,
    )

    print("\n===== TRAIN/TEST SPLIT =====")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=200)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Test accuracy: {accuracy:.4f}")

    # ---------------------------------------------------------
    # 2. Stratified Train/Test Split
    # ---------------------------------------------------------
    X_train_strat, X_test_strat, y_train_strat, y_test_strat = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=y,
        )
    )

    print("\n===== STRATIFIED TRAIN/TEST SPLIT =====")
    print(f"Training samples: {len(X_train_strat)}")
    print(f"Testing samples: {len(X_test_strat)}")

    print("Training class distribution:")
    print(np.bincount(y_train_strat))

    print("Testing class distribution:")
    print(np.bincount(y_test_strat))

    # ---------------------------------------------------------
    # 3. K-Fold Cross-Validation
    # ---------------------------------------------------------
    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    kfold_scores = cross_val_score(
        model,
        X,
        y,
        cv=kfold,
        scoring="accuracy",
    )

    print("\n===== K-FOLD CROSS-VALIDATION =====")
    print(f"Fold scores: {kfold_scores}")
    print(f"Mean accuracy: {kfold_scores.mean():.4f}")
    print(f"Standard deviation: {kfold_scores.std():.4f}")

    # ---------------------------------------------------------
    # 4. Stratified K-Fold Cross-Validation
    # ---------------------------------------------------------
    stratified_kfold = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    stratified_scores = cross_val_score(
        model,
        X,
        y,
        cv=stratified_kfold,
        scoring="accuracy",
    )

    print("\n===== STRATIFIED K-FOLD CROSS-VALIDATION =====")
    print(f"Fold scores: {stratified_scores}")
    print(f"Mean accuracy: {stratified_scores.mean():.4f}")
    print(f"Standard deviation: {stratified_scores.std():.4f}")

    # ---------------------------------------------------------
    # 5. Summary
    # ---------------------------------------------------------
    print("\n===== SUMMARY =====")
    print("Train/test split completed successfully.")
    print("Stratified train/test split completed successfully.")
    print("K-Fold cross-validation completed successfully.")
    print("Stratified K-Fold cross-validation completed successfully.")


if __name__ == "__main__":
    main()
