"""
W4D3 - Model Serialisation using joblib and pickle.

This script:
1. Loads the Iris dataset.
2. Splits the data into training and testing sets.
3. Trains a Logistic Regression model.
4. Saves the trained model using joblib.
5. Saves the trained model using pickle.
6. Loads both serialized models.
7. Verifies that loaded models make predictions.
"""

from pathlib import Path
import pickle

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_DIR = Path("saved_models")
JOBLIB_PATH = MODEL_DIR / "iris_logistic_regression.joblib"
PICKLE_PATH = MODEL_DIR / "iris_logistic_regression.pkl"


def train_model():
    """Train a Logistic Regression model on the Iris dataset."""
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=iris.target,
    )

    model = LogisticRegression(
        max_iter=200,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def save_with_joblib(model):
    """Save the trained model using joblib."""
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, JOBLIB_PATH)


def save_with_pickle(model):
    """Save the trained model using pickle."""
    MODEL_DIR.mkdir(exist_ok=True)

    with open(PICKLE_PATH, "wb") as file:
        pickle.dump(model, file)


def load_with_joblib():
    """Load the model saved with joblib."""
    return joblib.load(JOBLIB_PATH)


def load_with_pickle():
    """Load the model saved with pickle."""
    with open(PICKLE_PATH, "rb") as file:
        return pickle.load(file)


def main():
    """Train, serialize, deserialize, and evaluate the model."""
    model, X_test, y_test = train_model()

    original_predictions = model.predict(X_test)
    original_accuracy = accuracy_score(y_test, original_predictions)

    save_with_joblib(model)
    save_with_pickle(model)

    joblib_model = load_with_joblib()
    pickle_model = load_with_pickle()

    joblib_predictions = joblib_model.predict(X_test)
    pickle_predictions = pickle_model.predict(X_test)

    joblib_accuracy = accuracy_score(y_test, joblib_predictions)
    pickle_accuracy = accuracy_score(y_test, pickle_predictions)

    print("Model Serialisation Results")
    print("-" * 30)
    print(f"Original model accuracy: {original_accuracy:.4f}")
    print(f"Joblib model accuracy:    {joblib_accuracy:.4f}")
    print(f"Pickle model accuracy:    {pickle_accuracy:.4f}")
    print(f"Joblib file exists:       {JOBLIB_PATH.exists()}")
    print(f"Pickle file exists:       {PICKLE_PATH.exists()}")

    # Verify serialization preserved predictions.
    assert (original_predictions == joblib_predictions).all()
    assert (original_predictions == pickle_predictions).all()

    print("\nSerialization verification: PASSED")


if __name__ == "__main__":
    main()
    