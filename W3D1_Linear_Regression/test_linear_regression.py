
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_data():
    """Load the Diabetes dataset and create a train/test split."""
    diabetes = load_diabetes()

    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data,
        diabetes.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


def test_dataset_shape():
    """Verify the Diabetes dataset has the expected dimensions."""
    diabetes = load_diabetes()

    assert diabetes.data.shape == (442, 10)
    assert diabetes.target.shape == (442,)


def test_train_test_split():
    """Verify the train/test split sizes."""
    X_train, X_test, y_train, y_test = get_data()

    assert X_train.shape[0] == 353
    assert X_test.shape[0] == 89
    assert y_train.shape[0] == 353
    assert y_test.shape[0] == 89


def test_linear_regression_training():
    """Verify that Linear Regression can train and predict."""
    X_train, X_test, y_train, y_test = get_data()

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert predictions.shape == y_test.shape
    assert len(model.coef_) == 10


def test_regression_metrics():
    """Verify that all required regression metrics are valid."""
    X_train, X_test, y_train, y_test = get_data()

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    assert mse >= 0
    assert rmse >= 0
    assert mae >= 0
    assert -1 <= r2 <= 1


def test_ridge_and_lasso_training():
    """Verify that Ridge and Lasso models can train and predict."""
    X_train, X_test, y_train, y_test = get_data()

    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=1.0)

    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)

    ridge_predictions = ridge.predict(X_test)
    lasso_predictions = lasso.predict(X_test)

    assert ridge_predictions.shape == y_test.shape
    assert lasso_predictions.shape == y_test.shape


def test_three_models_are_evaluated():
    """Verify that all three regression models produce valid R² scores."""
    X_train, X_test, y_train, y_test = get_data()

    models = [
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=1.0)
    ]

    scores = []

    for model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores.append(r2_score(y_test, predictions))

    assert len(scores) == 3
    assert all(-1 <= score <= 1 for score in scores)
