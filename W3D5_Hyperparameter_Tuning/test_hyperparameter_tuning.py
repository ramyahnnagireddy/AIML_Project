"""
Tests for W3D5 Hyperparameter Tuning.
"""

from hyperparameter_tuning import (
    create_pipeline,
    load_data,
    run_grid_search,
    run_random_search,
)


def test_load_data():
    """Test that the dataset is split correctly."""
    X_train, X_test, y_train, y_test = load_data()

    assert len(X_train) == 120
    assert len(X_test) == 30
    assert len(y_train) == 120
    assert len(y_test) == 30


def test_create_pipeline():
    """Test that the SVM pipeline is created correctly."""
    pipeline = create_pipeline()

    assert "scaler" in pipeline.named_steps
    assert "svm" in pipeline.named_steps


def test_grid_search():
    """Test that GridSearchCV finds a valid model."""
    X_train, _, y_train, _ = load_data()

    grid_search = run_grid_search(X_train, y_train)

    assert grid_search.best_estimator_ is not None
    assert "svm__C" in grid_search.best_params_
    assert "svm__kernel" in grid_search.best_params_


def test_random_search():
    """Test that RandomizedSearchCV finds a valid model."""
    X_train, _, y_train, _ = load_data()

    random_search = run_random_search(X_train, y_train)

    assert random_search.best_estimator_ is not None
    assert "svm__C" in random_search.best_params_
    assert "svm__kernel" in random_search.best_params_
    