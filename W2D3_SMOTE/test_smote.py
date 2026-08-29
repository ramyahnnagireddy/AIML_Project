"""
Tests for W2D3 SMOTE implementation.
"""

import numpy as np

from smote_imbalanced_data import (
    apply_smote,
    create_dataset,
    split_data,
)


def test_dataset_is_imbalanced():
    """Verify that the generated dataset is imbalanced."""
    _, y = create_dataset()

    unique, counts = np.unique(y, return_counts=True)

    assert len(unique) == 2
    assert counts[0] != counts[1]


def test_smote_balances_training_data():
    """Verify that SMOTE balances the minority class."""
    X, y = create_dataset()

    X_train, _, y_train, _ = split_data(X, y)

    X_resampled, y_resampled = apply_smote(
        X_train,
        y_train,
    )

    unique, counts = np.unique(
        y_resampled,
        return_counts=True,
    )

    assert len(X_resampled) > len(X_train)
    assert counts[0] == counts[1]


def test_smote_preserves_feature_count():
    """Verify that SMOTE does not change the number of features."""
    X, y = create_dataset()

    X_train, _, y_train, _ = split_data(X, y)

    X_resampled, _ = apply_smote(
        X_train,
        y_train,
    )

    assert X_resampled.shape[1] == X_train.shape[1]