"""
Tests for W1D3 – Data Loading, Cleaning & Inspection.
"""

from io import StringIO

import pandas as pd

from data_loading_cleaning_inspection import clean_data


def test_clean_data():
    """Verify duplicate removal and missing-value cleaning."""

    csv_data = """Name,Age,City,Salary
Ramya,20,Bangalore,35000
Rahul,,Mysore,40000
Rahul,,Mysore,40000
Anu,22,,38000
"""

    df = pd.read_csv(StringIO(csv_data))
    cleaned_df = clean_data(df)

    # Duplicate row should be removed.
    assert len(cleaned_df) == 3
    assert cleaned_df.duplicated().sum() == 0

    # Missing numeric values should be filled.
    assert cleaned_df["Age"].isnull().sum() == 0
    assert cleaned_df["Salary"].isnull().sum() == 0

    # Missing categorical value should be filled.
    assert cleaned_df["City"].isnull().sum() == 0
    