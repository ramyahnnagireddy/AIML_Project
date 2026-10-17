"""
Tests for the W2D5 Titanic End-to-End Preprocessing Pipeline.
"""

import os
import pandas as pd


# Path to the exported ML-ready dataset.
OUTPUT_FILE = "W2D5_End_to_End_Preprocessing/titanic_processed.csv"


def test_output_file_exists():
    """Verify that the processed CSV file exists."""
    assert os.path.exists(OUTPUT_FILE)


def test_output_shape():
    """Verify the processed dataset has 891 rows and 22 columns."""
    df = pd.read_csv(OUTPUT_FILE)

    assert df.shape == (891, 22)


def test_no_missing_values():
    """Verify that the ML-ready dataset contains no missing values."""
    df = pd.read_csv(OUTPUT_FILE)

    assert df.isnull().sum().sum() == 0


def test_target_column_exists():
    """Verify that the target column exists."""
    df = pd.read_csv(OUTPUT_FILE)

    assert "survived" in df.columns


def test_target_distribution():
    """Verify that the Titanic target contains both classes."""
    df = pd.read_csv(OUTPUT_FILE)

    assert set(df["survived"].unique()) == {0, 1}