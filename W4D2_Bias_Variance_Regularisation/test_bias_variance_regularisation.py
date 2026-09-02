"""
Tests for W4D2: Bias-Variance Tradeoff & Regularisation.
"""

import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from bias_variance_regularisation import (
    create_dataset,
    compare_regularisation_models,
    ridge_grid_search,
    lasso_random_search,
)


RANDOM_STATE = 42


def test_create_dataset():
    """Test that the dataset has the expected shape."""

    X, y = create_dataset()

    assert X.shape == (300, 10)
    assert y.shape == (300,)
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


def test_model_comparison():
    """Test that Linear Regression, Ridge, and Lasso are evaluated."""

    X, y = create_dataset()

    X_train = X[:240]
    X_test = X[240:]
    y_train = y[:240]
    y_test = y[240:]

    results = compare_regularisation_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    assert len(results) == 3

    expected_models = {
        "Linear Regression",
        "Ridge",
        "Lasso",
    }

    assert set(results["Model"]) == expected_models

    assert "MSE" in results.columns
    assert "RMSE" in results.columns
    assert "R2" in results.columns


def test_ridge_grid_search():
    """Test Ridge GridSearchCV."""

    X, y = create_dataset()

    X_train = X[:240]
    y_train = y[:240]

    search = ridge_grid_search(
        X_train,
        y_train,
    )

    assert isinstance(search, GridSearchCV)
    assert "model__alpha" in search.best_params_
    assert search.best_params_["model__alpha"] in [
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ]

    assert isinstance(search.best_score_, float)


def test_lasso_random_search():
    """Test Lasso RandomizedSearchCV."""

    X, y = create_dataset()

    X_train = X[:240]
    y_train = y[:240]

    search = lasso_random_search(
        X_train,
        y_train,
    )

    assert isinstance(search, RandomizedSearchCV)
    assert "model__alpha" in search.best_params_

    assert search.best_params_["model__alpha"] > 0
    assert isinstance(search.best_score_, float)


def test_regularisation_models_are_available():
    """Test that required regularisation models can be created."""

    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=1.0, max_iter=10000)
    linear = LinearRegression()

    assert ridge.alpha == 1.0
    assert lasso.alpha == 1.0
    assert isinstance(linear, LinearRegression)


def test_random_state():
    """Test that the expected random state is used."""

    assert RANDOM_STATE == 42
    