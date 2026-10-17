import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


# -------------------------------------------------
# Configuration
# -------------------------------------------------

RANDOM_STATE = 42

OUTPUT_DIR = Path("W2D2_Feature_Engineering")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

np.random.seed(RANDOM_STATE)


# -------------------------------------------------
# 1. Create Sample Dataset
# -------------------------------------------------

data = {
    "Age": [22, 25, 28, 35, 40, 45, 30, 27, 50, 32],

    "Salary": [
        25000, 30000, 35000, 50000, 60000,
        70000, 42000, 38000, 80000, 45000
    ],

    "Experience": [
        1, 2, 3, 8, 12,
        15, 5, 4, 18, 6
    ],

    "City": [
        "Bangalore",
        "Chennai",
        "Bangalore",
        "Mumbai",
        "Delhi",
        "Chennai",
        "Delhi",
        "Mumbai",
        "Bangalore",
        "Chennai",
    ],

    "Education": [
        "Graduate",
        "Postgraduate",
        "Graduate",
        "Postgraduate",
        "PhD",
        "Graduate",
        "Postgraduate",
        "Graduate",
        "PhD",
        "Postgraduate",
    ],

    "Performance": [
        "Average",
        "Good",
        "Good",
        "Excellent",
        "Excellent",
        "Good",
        "Average",
        "Good",
        "Excellent",
        "Average",
    ],

    "Promoted": [
        0, 0, 1, 1, 1,
        1, 0, 1, 1, 0
    ],
}

df = pd.DataFrame(data)

print("===== ORIGINAL DATASET =====")
print(df)

print("\nDataset shape:", df.shape)


# -------------------------------------------------
# 2. Label Encoding
# -------------------------------------------------

label_encoder = LabelEncoder()

df["Performance_Label"] = label_encoder.fit_transform(
    df["Performance"]
)

performance_map = dict(
    zip(
        label_encoder.classes_,
        label_encoder.transform(label_encoder.classes_),
    )
)

print("\n===== LABEL ENCODER =====")
print(
    df[
        [
            "Performance",
            "Performance_Label"
        ]
    ]
)

print("Mapping:", performance_map)

# LabelEncoder converts categories into numeric labels.
# For unordered categories, OneHotEncoder is generally
# preferred because LabelEncoder can introduce an
# artificial numeric ordering.


# -------------------------------------------------
# 3. One-Hot Encoding
# -------------------------------------------------

onehot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore",
)

city_encoded = onehot_encoder.fit_transform(
    df[["City"]]
)

city_columns = onehot_encoder.get_feature_names_out(
    ["City"]
)

city_encoded_df = pd.DataFrame(
    city_encoded,
    columns=city_columns,
    index=df.index,
)

print("\n===== ONE HOT ENCODER =====")
print(city_encoded_df)

# OneHotEncoder is suitable for unordered categories
# such as City.
#
# handle_unknown="ignore" prevents errors if a new
# city appears during future prediction.


# -------------------------------------------------
# 4. Ordinal Encoding
# -------------------------------------------------

education_order = [
    [
        "Graduate",
        "Postgraduate",
        "PhD"
    ]
]

ordinal_encoder = OrdinalEncoder(
    categories=education_order
)

ordinal_values = ordinal_encoder.fit_transform(
    df[["Education"]]
)

df["Education_Ordinal"] = (
    ordinal_values
    .astype(int)
    .ravel()
)

print("\n===== ORDINAL ENCODER =====")

print(
    df[
        [
            "Education",
            "Education_Ordinal"
        ]
    ]
)

# Education has a meaningful order:
#
# Graduate < Postgraduate < PhD
#
# Therefore OrdinalEncoder is appropriate.


# -------------------------------------------------
# 5. Prepare Features and Target
# -------------------------------------------------

target = df["Promoted"]

numeric_features = df[
    [
        "Age",
        "Salary",
        "Experience"
    ]
]

X = pd.concat(
    [
        numeric_features,

        df[
            [
                "Performance_Label",
                "Education_Ordinal"
            ]
        ],

        city_encoded_df,
    ],
    axis=1,
)

y = target


# -------------------------------------------------
# 6. Train-Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n===== TRAIN TEST SPLIT =====")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# -------------------------------------------------
# 7. Feature Scaling
# -------------------------------------------------

# Only raw continuous numeric variables are scaled.
#
# Encoded categorical variables are NOT scaled here.

numeric_cols = [
    "Age",
    "Salary",
    "Experience"
]

X_train_numeric = X_train[numeric_cols]
X_test_numeric = X_test[numeric_cols]


# -------------------------------------------------
# 7.1 Create Scalers
# -------------------------------------------------

standard_scaler = StandardScaler()

minmax_scaler = MinMaxScaler()

robust_scaler = RobustScaler(
    quantile_range=(25.0, 75.0)
)


# -------------------------------------------------
# 7.2 Fit ONLY on Training Data
# -------------------------------------------------

standard_train = (
    standard_scaler.fit_transform(
        X_train_numeric
    )
)

standard_test = (
    standard_scaler.transform(
        X_test_numeric
    )
)


minmax_train = (
    minmax_scaler.fit_transform(
        X_train_numeric
    )
)

minmax_test = (
    minmax_scaler.transform(
        X_test_numeric
    )
)


robust_train = (
    robust_scaler.fit_transform(
        X_train_numeric
    )
)

robust_test = (
    robust_scaler.transform(
        X_test_numeric
    )
)


# -------------------------------------------------
# 7.3 Convert Scaled Arrays to DataFrames
# -------------------------------------------------

standard_train_df = pd.DataFrame(
    standard_train,
    columns=numeric_cols,
    index=X_train_numeric.index,
)

standard_test_df = pd.DataFrame(
    standard_test,
    columns=numeric_cols,
    index=X_test_numeric.index,
)


minmax_train_df = pd.DataFrame(
    minmax_train,
    columns=numeric_cols,
    index=X_train_numeric.index,
)

minmax_test_df = pd.DataFrame(
    minmax_test,
    columns=numeric_cols,
    index=X_test_numeric.index,
)


robust_train_df = pd.DataFrame(
    robust_train,
    columns=numeric_cols,
    index=X_train_numeric.index,
)

robust_test_df = pd.DataFrame(
    robust_test,
    columns=numeric_cols,
    index=X_test_numeric.index,
)


# -------------------------------------------------
# 7.4 Display Scaling Results
# -------------------------------------------------

print("\n===== STANDARDSCALER =====")
print(standard_train_df)


print("\n===== MINMAXSCALER =====")
print(minmax_train_df)


print("\n===== ROBUSTSCALER =====")
print(robust_train_df)


# -------------------------------------------------
# 7.5 Scaling Distribution Visualization
# -------------------------------------------------

salary_train = X_train_numeric["Salary"]

standard_salary = standard_train_df["Salary"]

minmax_salary = minmax_train_df["Salary"]

robust_salary = robust_train_df["Salary"]


fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 4),
)


# Original distribution

axes[0].hist(
    salary_train,
    bins=5,
)

axes[0].set_title(
    "Before Scaling - Salary"
)

axes[0].set_xlabel(
    "Salary"
)


# StandardScaler

axes[1].hist(
    standard_salary,
    bins=5,
)

axes[1].set_title(
    "StandardScaler - Salary"
)

axes[1].set_xlabel(
    "Scaled Salary"
)


# MinMaxScaler

axes[2].hist(
    minmax_salary,
    bins=5,
)

axes[2].set_title(
    "MinMaxScaler - Salary"
)

axes[2].set_xlabel(
    "Scaled Salary"
)


# RobustScaler

axes[3].hist(
    robust_salary,
    bins=5,
)

axes[3].set_title(
    "RobustScaler - Salary"
)

axes[3].set_xlabel(
    "Scaled Salary"
)


plt.tight_layout()

plot_path = (
    OUTPUT_DIR /
    "scaling_distributions.png"
)

plt.savefig(plot_path)

plt.close()

print(
    "\nScaling distribution plot saved to:",
    plot_path
)


# -------------------------------------------------
# 8. Remove Constant Features
# -------------------------------------------------

constant_features = X_train.columns[
    X_train.nunique() <= 1
]

print(
    "\n===== CONSTANT FEATURES REMOVED ====="
)

if len(constant_features) > 0:

    for feature in constant_features:

        print(
            "-",
            feature
        )

else:

    print("None")


X_train_filtered = X_train.drop(
    columns=constant_features
)

X_test_filtered = X_test.drop(
    columns=constant_features
)


# -------------------------------------------------
# 9. SelectKBest - Top 5 Features
# -------------------------------------------------

# SelectKBest is fitted ONLY on training data.
#
# This prevents information from the test set
# leaking into feature selection.

selector = SelectKBest(
    score_func=f_classif,
    k=min(
        5,
        X_train_filtered.shape[1]
    ),
)


X_train_selected = selector.fit_transform(
    X_train_filtered,
    y_train,
)


X_test_selected = selector.transform(
    X_test_filtered
)


selected_features = (
    X_train_filtered.columns[
        selector.get_support()
    ]
)


print(
    "\n===== SELECTKBEST TOP 5 FEATURES ====="
)

for feature in selected_features:

    print(
        "-",
        feature
    )


# -------------------------------------------------
# 10. Feature Scores
# -------------------------------------------------

feature_scores = pd.DataFrame(
    {
        "Feature": X_train_filtered.columns,
        "Score": selector.scores_,
    }
)

feature_scores = feature_scores.sort_values(
    by="Score",
    ascending=False,
)


print(
    "\nFeature scores:"
)

print(feature_scores)


# -------------------------------------------------
# 11. Why Selected Features Matter
# -------------------------------------------------

feature_explanations = {

    "Age":
        "May capture career stage and "
        "experience-related promotion patterns.",

    "Salary":
        "Can reflect responsibility level "
        "and job seniority.",

    "Experience":
        "Directly represents professional "
        "experience.",

    "Performance_Label":
        "Represents the employee's "
        "performance category.",

    "Education_Ordinal":
        "Represents the ordered education "
        "level.",

    "City_Bangalore":
        "Indicates whether the employee "
        "is from Bangalore.",

    "City_Chennai":
        "Indicates whether the employee "
        "is from Chennai.",

    "City_Delhi":
        "Indicates whether the employee "
        "is from Delhi.",

    "City_Mumbai":
        "Indicates whether the employee "
        "is from Mumbai.",
}


print(
    "\n===== WHY THESE FEATURES MATTER ====="
)


for feature in selected_features:

    explanation = feature_explanations.get(
        feature,
        "Selected based on its statistical score."
    )

    print(
        f"{feature}: {explanation}"
    )


# -------------------------------------------------
# 12. W2D2 Completion
# -------------------------------------------------

print(
    "\n===== W2D2 COMPLETE ====="
)
