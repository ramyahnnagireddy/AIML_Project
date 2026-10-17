"""
W3D5: Hyperparameter Tuning — GridSearchCV and RandomizedSearchCV

This script:
1. Loads the Iris dataset.
2. Splits the data into training and testing sets.
3. Builds an SVM pipeline with StandardScaler.
4. Tunes SVM hyperparameters using GridSearchCV.
5. Tunes SVM hyperparameters using RandomizedSearchCV.
6. Evaluates and compares both methods.
7. Logs results using MLflow.
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Constants for reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def load_data():
    """Load the Iris dataset and split it into train and test sets."""

    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=iris.target,
    )

    return X_train, X_test, y_train, y_test


def create_pipeline():
    """Create a machine learning pipeline with scaling and SVM."""

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC()),
        ]
    )

    return pipeline


def run_grid_search(X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""

    # GridSearchCV tests every possible combination.
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__kernel": ["linear", "rbf"],
        "svm__gamma": ["scale", "auto"],
    }

    grid_search = GridSearchCV(
        estimator=create_pipeline(),
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def run_random_search(X_train, y_train):
    """Perform hyperparameter tuning using RandomizedSearchCV."""

    # RandomizedSearchCV tests a random selection of combinations.
    param_distributions = {
        "svm__C": [0.01, 0.1, 1, 10, 100, 1000],
        "svm__kernel": ["linear", "rbf", "poly", "sigmoid"],
        "svm__gamma": ["scale", "auto"],
    }

    random_search = RandomizedSearchCV(
        estimator=create_pipeline(),
        param_distributions=param_distributions,
        n_iter=10,
        cv=CV_FOLDS,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)

    return random_search


def log_with_mlflow(model_name, search_result, X_test, y_test):
    """Log tuning results and the best model using MLflow."""

    test_accuracy = search_result.score(X_test, y_test)

    with mlflow.start_run(run_name=model_name):

        # Log basic information.
        mlflow.log_param("model", "SVM")
        mlflow.log_param("search_method", model_name)

        # Log the best hyperparameters.
        mlflow.log_params(
            {
                key: str(value)
                for key, value in search_result.best_params_.items()
            }
        )

        # Log evaluation metrics.
        mlflow.log_metric(
            "best_cv_accuracy",
            search_result.best_score_,
        )

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy,
        )

        # Log the best trained model.
        mlflow.sklearn.log_model(
            search_result.best_estimator_,
            name="best_model",
        )

    return test_accuracy


def main():
    """Run the complete hyperparameter tuning workflow."""

    # Load and split the dataset.
    X_train, X_test, y_train, y_test = load_data()

    print("Dataset loaded successfully.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Run GridSearchCV.
    print("\nRunning GridSearchCV...")
    grid_search = run_grid_search(X_train, y_train)

    # Run RandomizedSearchCV.
    print("Running RandomizedSearchCV...")
    random_search = run_random_search(X_train, y_train)

    # Log results using MLflow.
    grid_test_accuracy = log_with_mlflow(
        "GridSearchCV",
        grid_search,
        X_test,
        y_test,
    )

    random_test_accuracy = log_with_mlflow(
        "RandomizedSearchCV",
        random_search,
        X_test,
        y_test,
    )

    # Display GridSearchCV results.
    print("\n===== GridSearchCV Results =====")
    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Accuracy:", grid_search.best_score_)
    print("Test Accuracy:", grid_test_accuracy)

    # Display RandomizedSearchCV results.
    print("\n===== RandomizedSearchCV Results =====")
    print("Best Parameters:", random_search.best_params_)
    print("Best CV Accuracy:", random_search.best_score_)
    print("Test Accuracy:", random_test_accuracy)

    # Compare both tuning methods.
    print("\n===== Comparison =====")

    if grid_search.best_score_ >= random_search.best_score_:
        print("GridSearchCV achieved the best CV accuracy.")
    else:
        print("RandomizedSearchCV achieved the best CV accuracy.")
# Results:
# GridSearchCV: 97.5% CV accuracy, 93.33% test accuracy.
# RandomizedSearchCV: 97.5% CV accuracy, 93.33% test accuracy.
# Both methods selected a linear SVM with C=0.1.

if __name__ == "__main__":
    main()
