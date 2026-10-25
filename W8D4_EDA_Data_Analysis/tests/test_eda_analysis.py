"""Tests for the W8D4 exploratory data analysis workflow."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from eda_analysis import create_observations, load_dataset


def test_load_dataset_shape_and_columns():
    """The dataset should contain 569 records and 31 columns."""
    df = load_dataset()

    assert df.shape == (569, 31)
    assert "diagnosis" in df.columns


def test_dataset_has_no_missing_values():
    """All dataset fields should be populated."""
    df = load_dataset()

    assert int(df.isnull().sum().sum()) == 0


def test_diagnosis_categories():
    """Both expected diagnosis categories should be present."""
    df = load_dataset()

    assert set(df["diagnosis"].unique()) == {"benign", "malignant"}
    assert df["diagnosis"].value_counts().to_dict() == {
        "benign": 357,
        "malignant": 212,
    }


def test_create_observations_returns_five_items():
    """The EDA workflow should generate five data-based observations."""
    df = load_dataset()
    observations = create_observations(df)

    assert len(observations) == 5
    assert all(isinstance(item, str) and item for item in observations)


def test_numeric_features_are_numeric():
    """The 30 measurement columns should be numeric."""
    df = load_dataset()
    numeric_df = df.select_dtypes(include="number")

    assert numeric_df.shape == (569, 30)
    