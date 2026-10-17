
"""Exploratory Data Analysis for the Wisconsin diagnostic dataset."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = BASE_DIR / "plots"


def load_dataset() -> pd.DataFrame:
    """Load the built-in dataset and prepare readable class labels."""
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.data.copy()
    df["diagnosis"] = dataset.target.map(
        dict(enumerate(dataset.target_names))
    )
    return df


def create_observations(df: pd.DataFrame) -> list[str]:
    """Generate five observations directly from the dataset."""
    numeric = df.select_dtypes(include="number")
    diagnosis_counts = df["diagnosis"].value_counts()
    missing_total = int(df.isnull().sum().sum())
    feature_means = numeric.mean()
    feature_stds = numeric.std()

    return [
        f"The dataset contains {len(df)} rows and {df.shape[1]} columns.",
        (
            f"Missing values: {missing_total} across the entire dataset."
        ),
        (
            "Diagnosis distribution: "
            + ", ".join(
                f"{label}={count}"
                for label, count in diagnosis_counts.items()
            )
            + "."
        ),
        (
            f"The numeric feature with the largest mean is "
            f"'{feature_means.idxmax()}' "
            f"(mean={feature_means.max():.3f})."
        ),
        (
            f"The numeric feature with the largest standard deviation is "
            f"'{feature_stds.idxmax()}' "
            f"(standard deviation={feature_stds.max():.3f})."
        ),
    ]


def create_plots(df: pd.DataFrame) -> None:
    """Create feature distributions, correlation heatmap and class counts."""
    sns.set_theme(style="whitegrid")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    features = [
        "mean radius",
        "mean texture",
        "mean perimeter",
    ]

    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            data=df,
            x=feature,
            hue="diagnosis",
            kde=True,
            ax=ax,
        )
        ax.set_title(f"Distribution of {feature}")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"distribution_{feature.replace(' ', '_')}.png")
        plt.close(fig)

    numeric = df.select_dtypes(include="number")
    selected_features = numeric.var().nlargest(10).index
    correlation = numeric[selected_features].corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        annot=False,
        square=True,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap: 10 Highest-Variance Features")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "correlation_heatmap.png")
    plt.close(fig)

    counts = df["diagnosis"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title("Top Diagnosis Category Counts")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Number of records")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "top_10_categorical_counts.png")
    plt.close(fig)


def run_eda() -> pd.DataFrame:
    """Run the complete EDA workflow and save evidence files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    df.to_csv(OUTPUT_DIR / "dataset.csv", index=False)
    df.describe().to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

    missing = df.isnull().sum().rename("missing_count")
    missing.to_csv(OUTPUT_DIR / "missing_values.csv")

    info_lines = [
        "DATASET INFORMATION",
        "===================",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        "",
        "Column data types:",
        df.dtypes.to_string(),
        "",
        "DataFrame info:",
    ]
    import io

    info_buffer = io.StringIO()
    df.info(buf=info_buffer)
    info_lines.append(info_buffer.getvalue())
    (OUTPUT_DIR / "dataset_info.txt").write_text(
        "\n".join(info_lines),
        encoding="utf-8",
    )

    observations = create_observations(df)
    observation_text = "# EDA Observations\n\n" + "\n".join(
        f"{index}. {item}"
        for index, item in enumerate(observations, start=1)
    )
    (OUTPUT_DIR / "observations.md").write_text(
        observation_text + "\n",
        encoding="utf-8",
    )

    narrative = """
    This exploratory data analysis examines the Wisconsin diagnostic
    dataset using pandas, Matplotlib and Seaborn. The dataset contains
    numeric measurements derived from digitized images of breast mass
    samples, along with a diagnosis category. Descriptive statistics
    summarize the central tendency, spread and range of numeric features.
    DataFrame information and missing-value counts help assess data
    completeness and column types before further analysis.

    Histograms of mean radius, mean texture and mean perimeter show how
    these measurements are distributed across diagnosis categories.
    Their shapes and overlap provide an initial view of differences
    between the groups, but distributions alone cannot establish a
    reliable diagnostic rule. A correlation heatmap highlights linear
    relationships among ten numeric features selected by variance.
    Strong correlations may indicate related measurements or redundant
    information, which should be considered in later modelling.

    The category-count chart summarizes the available diagnosis labels
    and helps reveal whether the classes are balanced. Five observations
    are generated from the actual data rather than manually assumed.
    This analysis is descriptive, not a clinical assessment or a
    validated diagnostic model. Any later predictive modelling would
    require careful train-test separation, suitable evaluation metrics,
    bias assessment and independent validation. Overall, these outputs
    provide a reproducible starting point for understanding the dataset
    and planning subsequent machine-learning experiments.
    """
    (OUTPUT_DIR / "eda_narrative.md").write_text(
        "\n".join(line.strip() for line in narrative.strip().splitlines())
        + "\n",
        encoding="utf-8",
    )

    create_plots(df)

    print("EDA completed successfully.")
    print(f"Dataset shape: {df.shape}")
    print("\n=== df.info() ===")
    df.info()
    print("\n=== df.describe() ===")
    print(df.describe())
    print("\n=== df.isnull().sum() ===")
    print(df.isnull().sum())
    print("\n=== Five observations ===")
    for number, observation in enumerate(observations, start=1):
        print(f"{number}. {observation}")
    print(f"\nEvidence saved to: {OUTPUT_DIR}")
    print(f"Plots saved to: {PLOT_DIR}")

    return df


if __name__ == "__main__":
    run_eda()
    