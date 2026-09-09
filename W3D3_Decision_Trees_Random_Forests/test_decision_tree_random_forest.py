"""
Tests for W3D3 Decision Tree and Random Forest models.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_data():
    """Load and split the breast cancer dataset."""
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def test_decision_tree_training():
    """Check that the Decision Tree trains successfully."""
    X_train, X_test, y_train, y_test = get_data()

    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_tuned_decision_tree_depth():
    """Check that the tuned tree has the expected depth limit."""
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_split=5,
        random_state=RANDOM_STATE,
    )

    assert model.max_depth == 4
    assert model.min_samples_split == 5


def test_random_forest_training():
    """Check that the Random Forest trains successfully."""
    X_train, X_test, y_train, y_test = get_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_random_forest_estimators():
    """Check that Random Forest uses 100 trees."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    )

    assert model.n_estimators == 100
    