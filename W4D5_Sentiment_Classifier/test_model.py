import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_data():
    data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target
    )

    return X_train, X_test, y_train, y_test


def test_logistic_regression_training():
    X_train, X_test, y_train, y_test = get_data()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.90


def test_random_forest_training():
    X_train, X_test, y_train, y_test = get_data()

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.90


def test_logistic_regression_roc_auc():
    X_train, X_test, y_train, y_test = get_data()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ])

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probabilities)

    assert roc_auc >= 0.90


def test_predictions_are_binary():
    X_train, X_test, y_train, y_test = get_data()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert set(np.unique(predictions)).issubset({0, 1})
def test_logistic_regression_precision_recall():
    X_train, X_test, y_train, y_test = get_data()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    assert precision >= 0.90
    assert recall >= 0.90


def test_random_forest_precision_recall():
    X_train, X_test, y_train, y_test = get_data()

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    assert precision >= 0.90
    assert recall >= 0.90
    