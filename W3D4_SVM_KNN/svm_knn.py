"""
W3D4: SVM & KNN — When to Use What

This script:
1. Loads the Iris dataset.
2. Splits the data into training and testing sets.
3. Scales the features.
4. Trains an SVM classifier.
5. Trains a KNN classifier.
6. Evaluates both models.
7. Compares their performance.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_and_prepare_data():
    """Load Iris dataset, split it, and scale the features."""
    iris = load_iris()

    X = iris.data
    y = iris.target

    # Split the dataset into training and testing data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Standardize features because SVM and KNN are
    # sensitive to differences in feature scale.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train the model and calculate evaluation metrics."""
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    return results


def main():
    """Train and compare SVM and KNN models."""

    # Prepare the dataset.
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Create the SVM model.
    svm_model = SVC(
        kernel="rbf",
        C=1.0,
        random_state=RANDOM_STATE,
    )

    # Create the KNN model.
    knn_model = KNeighborsClassifier(
        n_neighbors=5
    )

    # Evaluate SVM.
    svm_results = evaluate_model(
        svm_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Evaluate KNN.
    knn_results = evaluate_model(
        knn_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Display comparison results.
    print("\nSVM & KNN Model Comparison")
    print("=" * 45)

    print("\nSVM Results:")
    for metric, value in svm_results.items():
        print(f"{metric.capitalize():<12}: {value:.4f}")

    print("\nKNN Results:")
    for metric, value in knn_results.items():
        print(f"{metric.capitalize():<12}: {value:.4f}")

    # Compare models using accuracy.
    if svm_results["accuracy"] > knn_results["accuracy"]:
        better_model = "SVM"
    elif knn_results["accuracy"] > svm_results["accuracy"]:
        better_model = "KNN"
    else:
        better_model = "Both models have the same accuracy"

    print("\n" + "=" * 45)
    print(f"Better model based on accuracy: {better_model}")


if __name__ == "__main__":
    main()
    