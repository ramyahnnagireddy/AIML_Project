"""
W1D4 – Exploratory Data Analysis (EDA)

Purpose:
    Perform basic exploratory data analysis on the Iris dataset.

Inputs:
    data/iris.csv

Outputs:
    EDA observations, narrative, and visualization images in outputs/.
"""

from pathlib import Path
import logging

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Project paths
DATA_PATH = Path("data/iris.csv")
OUTPUT_DIR = Path("outputs")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


def main():
    """Run the complete EDA workflow."""

    # Create output directory if it does not exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check dataset
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("===== DATASET SHAPE =====")
    print(df.shape)

    print("\n===== FIRST 10 ROWS =====")
    print(df.head(10))

    print("\n===== DESCRIPTIVE STATISTICS =====")
    print(df.describe())

    print("\n===== DATASET INFORMATION =====")
    df.info()

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    # ===== FIVE EDA OBSERVATIONS =====
    observations = [
        "1. The dataset contains 150 rows and 5 columns.",
        "2. There are four numerical features: sepal_length, sepal_width, "
        "petal_length, and petal_width.",
        "3. The species column is categorical and contains the flower species labels.",
        "4. There are no missing values in any of the five columns.",
        "5. The three species are evenly represented with 50 samples each, "
        "indicating a balanced target distribution."
    ]

    print("\n===== FIVE EDA OBSERVATIONS =====")
    for observation in observations:
        print(observation)

    # Numeric columns
    numeric_columns = df.select_dtypes(include="number").columns

    # Plot style
    sns.set_style("whitegrid")

    # ===== 1. NUMERIC DISTRIBUTIONS =====
    df[numeric_columns].hist(
        figsize=(10, 8),
        bins=15
    )

    plt.suptitle("Distribution of Numeric Features")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "numeric_distributions.png",
        bbox_inches="tight"
    )
    plt.close("all")

    # ===== 2. CORRELATION HEATMAP =====
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        df[numeric_columns].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        bbox_inches="tight"
    )
    plt.close(fig)

    # ===== 3. TOP-10 CATEGORY COUNTS =====
    fig, ax = plt.subplots(figsize=(8, 5))

    df["species"].value_counts().head(10).plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Top-10 Species Counts")
    ax.set_xlabel("Species")
    ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "top10_species_counts.png",
        bbox_inches="tight"
    )
    plt.close(fig)

    logging.info("EDA plots saved successfully.")

    # ===== EDA NARRATIVE =====
    eda_narrative = """
The Iris dataset contains 150 observations and five columns, including four
numerical measurements and one categorical species column. The numerical
features are sepal length, sepal width, petal length, and petal width. The
descriptive statistics show that the features have different ranges and
levels of variation. Petal length and petal width show noticeable variation
across the dataset, while sepal width has a comparatively smaller range.

The distribution plots show that some numerical features are not perfectly
uniform and may contain multiple clusters. This is expected because the
dataset contains different flower species with different measurements. The
correlation analysis indicates strong relationships between some petal
measurements, which is useful for understanding feature relationships.

No missing values were found in any column, so missing-value treatment is not
required. There are also no obvious data-format problems because the four
measurement columns are numeric and the species column is categorical.

One point that should be examined further is the presence of different
clusters and potentially unusual observations in the distributions. These
should be checked before applying machine learning models. The dataset is
already relatively clean, but feature scaling may be useful for algorithms
that are sensitive to feature magnitude. Overall, the data is suitable for
further analysis and machine learning after checking distributions,
correlations, and possible outliers.
"""

    print("\n===== EDA NARRATIVE =====")
    print(eda_narrative.strip())

    logging.info("EDA analysis completed successfully.")


if __name__ == "__main__":
    main()
    