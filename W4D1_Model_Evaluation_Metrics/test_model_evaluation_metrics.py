"""Tests for W4D1 model evaluation metrics."""

import numpy as np

from model_evaluation_metrics import (
    calculate_metrics,
    load_data,
    perform_stratified_cv,
    train_model,
)


def test_load_data():
    """Test that the dataset loads with expected dimensions."""
    X, y = load_data()

    assert X.shape == (569, 30)
    assert y.shape == (569,)
    assert len(np.unique(y)) == 2


def test_train_model():
    """Test that the model can be trained and make predictions."""
    X, y = load_data()

    model = train_model(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_calculate_metrics():
    """Test that all evaluation metrics are calculated."""
    X, y = load_data()

    model = train_model(X, y)
    metrics, matrix = calculate_metrics(model, X, y)

    expected_metrics = {
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
    }

    assert set(metrics.keys()) == expected_metrics

    for value in metrics.values():
        assert 0.0 <= value <= 1.0

    assert matrix.shape == (2, 2)


def test_stratified_cross_validation():
    """Test that Stratified K-Fold CV returns five scores."""
    X, y = load_data()

    results = perform_stratified_cv(X, y)

    expected_metrics = {
        "Accuracy",
        "Precision",
        "Recall",
        "ROC-AUC",
    }

    assert set(results.keys()) == expected_metrics

    for scores in results.values():
        assert len(scores) == 5
        assert np.all((scores >= 0.0) & (scores <= 1.0))
        