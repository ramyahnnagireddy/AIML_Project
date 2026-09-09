"""
W4D2: Bias-Variance Tradeoff & Regularisation

Topics covered:
- Bias and variance
- Linear Regression
- Ridge Regression (L2 regularisation)
- Lasso Regression (L1 regularisation)
- GridSearchCV
- RandomizedSearchCV
"""

import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def create_dataset():
    """Create a synthetic regression dataset."""

    X, y = make_regression(
        n_samples=300,
        n_features=10,
        noise=20,
        random_state=RANDOM_STATE,
    )

    return X, y


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train a model and return evaluation metrics."""

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    return mse, rmse, r2


def compare_regularisation_models(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Compare Linear Regression, Ridge, and Lasso."""

    models = {
        "Linear Regression": LinearRegression(),

        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),

        "Lasso": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=1.0, max_iter=10000)),
            ]
        ),
    }

    results = []

    for model_name, model in models.items():

        mse, rmse, r2 = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(
            {
                "Model": model_name,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    return pd.DataFrame(results)


def ridge_grid_search(X_train, y_train):
    """Find the best Ridge alpha using GridSearchCV."""

    ridge_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ]
    )

    parameter_grid = {
        "model__alpha": [
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ]
    }

    grid_search = GridSearchCV(
        estimator=ridge_pipeline,
        param_grid=parameter_grid,
        cv=CV_FOLDS,
        scoring="r2",
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def lasso_random_search(X_train, y_train):
    """Find the best Lasso alpha using RandomizedSearchCV."""

    lasso_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Lasso(max_iter=10000)),
        ]
    )

    parameter_distributions = {
        "model__alpha": np.logspace(-3, 2, 100)
    }

    random_search = RandomizedSearchCV(
        estimator=lasso_pipeline,
        param_distributions=parameter_distributions,
        n_iter=10,
        cv=CV_FOLDS,
        scoring="r2",
        random_state=RANDOM_STATE,
    )

    random_search.fit(X_train, y_train)

    return random_search


def main():
    """Run the complete W4D2 experiment."""

    # ---------------------------------------------------------
    # Step 1: Create dataset
    # ---------------------------------------------------------

    X, y = create_dataset()

    print("=" * 60)
    print("W4D2: BIAS-VARIANCE TRADEOFF & REGULARISATION")
    print("=" * 60)

    print("\nDataset shape:")
    print("Features:", X.shape)
    print("Target:", y.shape)

    # ---------------------------------------------------------
    # Step 2: Train/Test Split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("\nTrain/Test Split:")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ---------------------------------------------------------
    # Step 3: Compare models
    # ---------------------------------------------------------

    results = compare_regularisation_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(results.to_string(index=False))

    # ---------------------------------------------------------
    # Step 4: Ridge Grid Search
    # ---------------------------------------------------------

    ridge_search = ridge_grid_search(
        X_train,
        y_train,
    )

    print("\n" + "=" * 60)
    print("RIDGE - GRID SEARCH")
    print("=" * 60)

    print("Best parameters:")
    print(ridge_search.best_params_)

    print("Best CV R2:")
    print(round(ridge_search.best_score_, 4))

    ridge_predictions = ridge_search.predict(X_test)

    ridge_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            ridge_predictions,
        )
    )

    ridge_r2 = r2_score(
        y_test,
        ridge_predictions,
    )

    print("Test RMSE:", round(ridge_rmse, 4))
    print("Test R2:", round(ridge_r2, 4))

    # ---------------------------------------------------------
    # Step 5: Lasso Randomized Search
    # ---------------------------------------------------------

    lasso_search = lasso_random_search(
        X_train,
        y_train,
    )

    print("\n" + "=" * 60)
    print("LASSO - RANDOMIZED SEARCH")
    print("=" * 60)

    print("Best parameters:")
    print(lasso_search.best_params_)

    print("Best CV R2:")
    print(round(lasso_search.best_score_, 4))

    lasso_predictions = lasso_search.predict(X_test)

    lasso_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            lasso_predictions,
        )
    )

    lasso_r2 = r2_score(
        y_test,
        lasso_predictions,
    )

    print("Test RMSE:", round(lasso_rmse, 4))
    print("Test R2:", round(lasso_r2, 4))

    # ---------------------------------------------------------
    # Step 6: Concepts
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("CONCEPT SUMMARY")
    print("=" * 60)

    print("Linear Regression: No regularisation.")
    print("Ridge: L2 regularisation.")
    print("Lasso: L1 regularisation.")
    print("GridSearchCV: Tests every specified combination.")
    print("RandomizedSearchCV: Tests randomly selected combinations.")
    print("High bias: Usually indicates underfitting.")
    print("High variance: Usually indicates overfitting.")


if __name__ == "__main__":
    main()
    