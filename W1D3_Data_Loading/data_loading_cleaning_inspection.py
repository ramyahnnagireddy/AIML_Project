"""
W1D3 – Data Loading, Cleaning & Inspection

Author: Ramya HN
Internship: Cynaris AI/ML
"""

from pathlib import Path
import sys

import pandas as pd


def load_data(file_path: Path) -> pd.DataFrame:
    """Load the CSV file and raise a clear error if it is missing."""
    if not file_path.is_file():
        sys.exit(f"[ERROR] Data file not found: {file_path}")

    return pd.read_csv(file_path)


def inspect_data(df: pd.DataFrame, title: str) -> None:
    """Display basic information about the dataset."""
    print(f"\n=== {title} ===")

    print("\n=== DATASET SHAPE ===")
    print(df.shape)

    print("\n=== COLUMN NAMES ===")
    print(df.columns.tolist())

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    print("\n=== DUPLICATE ROWS ===")
    print(df.duplicated().sum())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and fill missing values."""
    df = df.drop_duplicates().copy()

    # Fill missing Age with the median.
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    # Fill missing City with the most frequent city.
    if "City" in df.columns:
        df["City"] = df["City"].fillna(df["City"].mode()[0])

    # Fill missing Salary with the median.
    if "Salary" in df.columns:
        df["Salary"] = df["Salary"].fillna(df["Salary"].median())

    return df


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save the cleaned DataFrame as a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved to: {output_path}")


def main() -> None:
    """Run the complete data loading and cleaning workflow."""

    # 1. Load the original dataset.
    input_path = Path("W1D3_Data_Loading/data/sample_data.csv")
    df = load_data(input_path)

    # 2. Inspect the original dataset.
    inspect_data(df, "ORIGINAL DATA")

    # 3. Clean the dataset.
    cleaned_df = clean_data(df)

    # 4. Inspect the cleaned dataset.
    inspect_data(cleaned_df, "CLEANED DATA")

    # 5. Save the cleaned dataset.
    output_path = Path(
        "W1D3_Data_Loading/data/cleaned_sample_data.csv"
    )
    save_data(cleaned_df, output_path)


if __name__ == "__main__":
    main()
    