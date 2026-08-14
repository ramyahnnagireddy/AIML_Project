"""
W1D5 - Data Visualisation using Matplotlib & Seaborn

Author: Ramya HN
Internship: Cynaris
Week 1 - Day 5

Purpose:
Create visualisations to understand distributions,
correlations, and trends in the dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create output directory if it does not exist
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load the Iris dataset
df = pd.read_csv("data/iris.csv")

# Display basic dataset information
print("===== DATASET SHAPE =====")
print(df.shape)

print("\n===== DATASET HEAD =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())
# ---------------------------------------------------------
# 1. Distribution of Numeric Features
# ---------------------------------------------------------

# Automatically identify numeric columns
numeric_columns = df.select_dtypes(include="number").columns.tolist()

plt.figure(figsize=(10, 6))

for column in numeric_columns:
    sns.histplot(
        data=df,
        x=column,
        kde=True,
        label=column,
        alpha=0.5
    )

plt.title("Distribution of Iris Numeric Features")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/feature_distributions.png")
plt.show()
plt.close()
# ---------------------------------------------------------
# 2. Correlation Heatmap
# ---------------------------------------------------------

correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Iris Features")
plt.tight_layout()

plt.savefig("outputs/visualisation_correlation_heatmap.png")
plt.show()
# ---------------------------------------------------------
# 3. Feature Comparison by Species
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="species",
    y="petal_length"
)

plt.title("Petal Length Distribution by Iris Species")
plt.xlabel("Species")
plt.ylabel("Petal Length")
plt.tight_layout()

plt.savefig("outputs/petal_length_by_species.png")
plt.show()
# ---------------------------------------------------------
# Visualisation Summary
# ---------------------------------------------------------

print("\n===== VISUALISATION SUMMARY =====")
print("1. Feature distributions saved to: outputs/feature_distributions.png")
print("2. Correlation heatmap saved to: outputs/visualisation_correlation_heatmap.png")
print("3. Species comparison saved to: outputs/petal_length_by_species.png")
# ---------------------------------------------------------
# Output File Tests
# ---------------------------------------------------------

assert Path("outputs/feature_distributions.png").exists()
assert Path("outputs/visualisation_correlation_heatmap.png").exists()
assert Path("outputs/petal_length_by_species.png").exists()

print("\n===== OUTPUT TESTS =====")
print("All visualisation output files exist.")
