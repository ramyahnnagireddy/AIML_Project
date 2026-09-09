"""
Tests for W4D3 model serialisation using joblib and pickle.
"""

from pathlib import Path

import joblib
import pickle

from model_serialisation import (
    JOBLIB_PATH,
    PICKLE_PATH,
    save_with_joblib,
    save_with_pickle,
    train_model,
    load_with_joblib,
    load_with_pickle,
)


def test_model_training():
    """Test that the Logistic Regression model trains successfully."""
    model, X_test, y_test = train_model()

    assert model is not None
    assert len(X_test) == len(y_test)


def test_joblib_serialisation():
    """Test saving and loading the model with joblib."""
    model, X_test, _ = train_model()

    save_with_joblib(model)

    assert Path(JOBLIB_PATH).exists()

    loaded_model = load_with_joblib()
    predictions = loaded_model.predict(X_test)

    assert len(predictions) == len(X_test)


def test_pickle_serialisation():
    """Test saving and loading the model with pickle."""
    model, X_test, _ = train_model()

    save_with_pickle(model)

    assert Path(PICKLE_PATH).exists()

    loaded_model = load_with_pickle()
    predictions = loaded_model.predict(X_test)

    assert len(predictions) == len(X_test)
    