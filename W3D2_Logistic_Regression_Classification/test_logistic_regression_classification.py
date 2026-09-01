"""
Tests for W3D2 Logistic Regression & Classification.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def create_model():
    """Create and train a Logistic Regression model on Iris data."""
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = LogisticRegression(
        max_iter=200,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def test_model_is_trained():
    """Verify that the Logistic Regression model is trained."""
    model, _, _ = create_model()

    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")


def test_predictions_have_correct_shape():
    """Verify that predictions match the test dataset size."""
    model, X_test, y_test = create_model()

    predictions = model.predict(X_test)

    assert predictions.shape == y_test.shape


def test_accuracy_is_reasonable():
    """Verify that the model achieves good classification accuracy."""
    model, X_test, y_test = create_model()

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.90


def test_probability_output():
    """Verify that class probabilities are valid."""
    model, X_test, _ = create_model()

    probabilities = model.predict_proba(X_test)

    assert probabilities.shape == (30, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    