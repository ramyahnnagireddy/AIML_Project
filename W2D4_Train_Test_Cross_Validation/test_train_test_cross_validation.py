"""
Tests for W2D4: Train/Test Split & Cross-Validation.
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


def test_train_test_split_sizes():
    """Verify that train/test split produces the expected sizes."""

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    assert len(X_train) == 120
    assert len(X_test) == 30
    assert len(y_train) == 120
    assert len(y_test) == 30


def test_stratified_split_preserves_class_distribution():
    """Verify that stratification preserves class proportions."""

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    assert np.bincount(y_train).tolist() == [40, 40, 40]
    assert np.bincount(y_test).tolist() == [10, 10, 10]


def test_kfold_returns_five_scores():
    """Verify that 5-fold cross-validation returns five scores."""

    iris = load_iris()
    X = iris.data
    y = iris.target

    model = LogisticRegression(max_iter=200)

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=kfold,
        scoring="accuracy",
    )

    assert len(scores) == 5
    assert all(0 <= score <= 1 for score in scores)


def test_stratified_kfold_returns_five_scores():
    """Verify that StratifiedKFold returns five valid scores."""

    iris = load_iris()
    X = iris.data
    y = iris.target

    model = LogisticRegression(max_iter=200)

    stratified_kfold = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=stratified_kfold,
        scoring="accuracy",
    )

    assert len(scores) == 5
    assert all(0 <= score <= 1 for score in scores)
    