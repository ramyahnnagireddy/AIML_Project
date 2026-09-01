"""
W2D1 - Feature Engineering & Encoding
Author: Ramya HN
Internship: Cynaris

Tasks:
1. Apply categorical encoding techniques.
2. Apply feature scaling techniques and visualize distributions.
3. Engineer features and identify the top 5 using SelectKBest.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

DATA_PATH = Path("data") / "iris.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

assert df.shape[0] > 0, "Dataset is empty."

print("===== ORIGINAL DATASET =====")
print(df.head())
print("\nDataset shape:", df.shape)


# ---------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------
# Create engineered features BEFORE scaling.

df["sepal_area"] = (
    df["sepal_length"] * df["sepal_width"]
)

df["petal_area"] = (
    df["petal_length"] * df["petal_width"]
)

df["sepal_aspect_ratio"] = (
    df["sepal_length"] / df["sepal_width"]
)

df["sepal_length_width_difference"] = (
    df["sepal_length"] - df["sepal_width"]
)

print("\n===== ENGINEERED FEATURES =====")
print(
    df[
        [
            "sepal_area",
            "petal_area",
            "sepal_aspect_ratio",
            "sepal_length_width_difference",
        ]
    ].head()
)


# ---------------------------------------------------------
# 3. Categorical Encoding
# ---------------------------------------------------------

# LabelEncoder:
# Simple integer mapping. Useful for target labels,
# but integer values may imply an artificial order.

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["species"])

print("\n===== LABEL ENCODER =====")
print(
    pd.DataFrame(
        {
            "species": df["species"].head(),
            "species_label": y[:5],
        }
    )
)

print("\nLabel mapping:")
for label, value in zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_),
):
    print(f"{label} -> {value}")


# OneHotEncoder:
# Creates separate binary columns and avoids artificial
# ordering, but can increase the number of features.

one_hot_encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = one_hot_encoder.fit_transform(
    df[["species"]]
)

one_hot_df = pd.DataFrame(
    one_hot_encoded,
    columns=one_hot_encoder.get_feature_names_out(["species"]),
)

print("\n===== ONE-HOT ENCODER =====")
print(one_hot_df.head())


# OrdinalEncoder:
# Appropriate when categories have a meaningful order.
# An incorrect order can introduce misleading relationships.

ordinal_encoder = OrdinalEncoder(
    categories=[["setosa", "versicolor", "virginica"]]
)

ordinal_encoded = ordinal_encoder.fit_transform(
    df[["species"]]
)

print("\n===== ORDINAL ENCODER =====")
print(
    pd.DataFrame(
        {
            "species": df["species"].head(),
            "species_ordinal": ordinal_encoded[:5, 0],
        }
    )
)

print("\nOrdinal categories:")
print(ordinal_encoder.categories_)


print("\n===== ENCODING TRADE-OFFS =====")
print("""
LabelEncoder:
- Simple integer mapping.
- Commonly appropriate for target labels.
- May imply an artificial order when used on nominal features.

OneHotEncoder:
- Creates separate binary columns.
- Does not introduce artificial ordering.
- Can increase the number of features with many categories.

OrdinalEncoder:
- Represents categories using ordered numerical values.
- Suitable when categories have a meaningful order.
- Incorrect ordering can introduce misleading relationships.
""")


# ---------------------------------------------------------
# 4. Feature Scaling
# ---------------------------------------------------------

numeric_columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "sepal_area",
    "petal_area",
    "sepal_aspect_ratio",
    "sepal_length_width_difference",
]

X_numeric = df[numeric_columns]

print("\n===== NUMERIC FEATURES BEFORE SCALING =====")
print(X_numeric.head())


# StandardScaler
standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(X_numeric)

standard_df = pd.DataFrame(
    standard_scaled,
    columns=[f"{column}_std" for column in numeric_columns],
)

print("\n===== STANDARDSCALER =====")
print(standard_df.head())


# MinMaxScaler
minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(X_numeric)

minmax_df = pd.DataFrame(
    minmax_scaled,
    columns=[f"{column}_minmax" for column in numeric_columns],
)

print("\n===== MINMAXSCALER =====")
print(minmax_df.head())


# RobustScaler
robust_scaler = RobustScaler()
robust_scaled = robust_scaler.fit_transform(X_numeric)

robust_df = pd.DataFrame(
    robust_scaled,
    columns=[f"{column}_robust" for column in numeric_columns],
)

print("\n===== ROBUSTSCALER =====")
print(robust_df.head())


# ---------------------------------------------------------
# 5. Distribution Plots - Before and After Scaling
# ---------------------------------------------------------
# Use sepal_length as a representative feature.

representative_column = "sepal_length"
column_index = numeric_columns.index(representative_column)

original_values = X_numeric[representative_column]
standard_values = standard_scaled[:, column_index]
minmax_values = minmax_scaled[:, column_index]
robust_values = robust_scaled[:, column_index]


def save_before_after_plot(
    original,
    scaled,
    scaler_name,
    output_file,
):
    """Save a before-and-after distribution comparison."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(original, bins=15, alpha=0.7)
    axes[0].set_title(
        f"Before Scaling - {representative_column}"
    )
    axes[0].set_xlabel(representative_column)
    axes[0].set_ylabel("Frequency")

    axes[1].hist(scaled, bins=15, alpha=0.7)
    axes[1].set_title(
        f"{scaler_name} - {representative_column}"
    )
    axes[1].set_xlabel("Scaled value")
    axes[1].set_ylabel("Frequency")

    fig.suptitle(
        f"{scaler_name}: Before vs After Scaling"
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_file, dpi=300)
    plt.close()


save_before_after_plot(
    original_values,
    standard_values,
    "StandardScaler",
    "scaling_standard.png",
)

save_before_after_plot(
    original_values,
    minmax_values,
    "MinMaxScaler",
    "scaling_minmax.png",
)

save_before_after_plot(
    original_values,
    robust_values,
    "RobustScaler",
    "scaling_robust.png",
)

# Separate original distribution evidence
plt.figure(figsize=(7, 5))
plt.hist(original_values, bins=15, alpha=0.7)
plt.title("Original Distribution - Sepal Length")
plt.xlabel("Sepal length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "scaling_before.png",
    dpi=300,
)
plt.close()

print("\nScaling plots saved:")
print("- outputs/scaling_before.png")
print("- outputs/scaling_standard.png")
print("- outputs/scaling_minmax.png")
print("- outputs/scaling_robust.png")


# ---------------------------------------------------------
# 6. SelectKBest Feature Selection
# ---------------------------------------------------------
# SelectKBest is applied AFTER scaling.

selector = SelectKBest(
    score_func=f_classif,
    k=5,
)

X_selected = selector.fit_transform(
    standard_scaled,
    y,
)

selected_mask = selector.get_support()

selected_features = [
    feature
    for feature, selected in zip(
        numeric_columns,
        selected_mask,
    )
    if selected
]

selected_scores = selector.scores_[selected_mask]

feature_scores = pd.DataFrame(
    {
        "Feature": selected_features,
        "F_Score": selected_scores,
    }
).sort_values(
    by="F_Score",
    ascending=False,
)


print("\n===== SELECTKBEST TOP 5 FEATURES =====")
print(feature_scores.to_string(index=False))


feature_reasons = {
    "petal_length":
        "Strongly separates Iris species because petal lengths differ substantially.",
    "petal_width":
        "Strongly distinguishes species and captures differences in petal size.",
    "petal_area":
        "Combines petal length and width to represent overall petal size.",
    "sepal_area":
        "Combines sepal length and width to represent overall sepal size.",
    "sepal_aspect_ratio":
        "Describes the relationship between sepal length and width.",
    "sepal_length_width_difference":
        "Measures the difference between sepal length and width.",
    "sepal_width":
        "Captures variation in sepal width between species.",
    "sepal_length":
        "Captures variation in sepal length between species.",
}


print("\nSelected features:")
for feature in selected_features:
    print(f"- {feature}")


print("\nWhy the selected features matter:")
for feature in feature_scores["Feature"]:
    print(
        f"- {feature}: "
        f"{feature_reasons.get(feature, 'Useful predictive feature.')}"
    )


print("\n===== W2D1 FEATURE ENGINEERING COMPLETE =====")
