"""
Tests for W3D4 SVM & KNN implementation.
"""

from svm_knn import load_and_prepare_data, evaluate_model
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def test_data_split():
    """Test that the data is split into expected sizes."""
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    assert X_train.shape[0] == 120
    assert X_test.shape[0] == 30
    assert len(y_train) == 120
    assert len(y_test) == 30


def test_svm_model():
    """Test that SVM produces valid evaluation metrics."""
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    model = SVC(kernel="rbf", C=1.0, random_state=42)

    results = evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["precision"] <= 1.0
    assert 0.0 <= results["recall"] <= 1.0
    assert 0.0 <= results["f1_score"] <= 1.0


def test_knn_model():
    """Test that KNN produces valid evaluation metrics."""
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    model = KNeighborsClassifier(n_neighbors=5)

    results = evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["precision"] <= 1.0
    assert 0.0 <= results["recall"] <= 1.0
    assert 0.0 <= results["f1_score"] <= 1.0
    