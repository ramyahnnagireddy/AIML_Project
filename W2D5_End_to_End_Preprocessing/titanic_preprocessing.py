"""
W2D5: End-to-End Preprocessing Pipeline
Titanic Dataset

Pipeline:
1. Load Titanic dataset
2. Perform EDA
3. Handle missing values
4. Encode categorical features
5. Scale numerical features
6. Export ML-ready features
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# ---------------------------------------------------------
# 1. LOAD TITANIC DATASET
# ---------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/"
    "master/titanic.csv"
)

df = pd.read_csv(DATA_URL)


# ---------------------------------------------------------
# 2. BASIC EDA
# ---------------------------------------------------------

print("=" * 60)
print("TITANIC DATASET - BASIC EDA")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values before preprocessing:")
print(df.isnull().sum())


# ---------------------------------------------------------
# 3. REMOVE UNNECESSARY / LEAKAGE COLUMNS
# ---------------------------------------------------------

# 'alive' directly represents the target 'survived'.
# Keeping it would cause target leakage.
# 'deck' has too many missing values, so it is removed.
df = df.drop(columns=["alive", "deck"])


# ---------------------------------------------------------
# 4. HANDLE MISSING VALUES
# ---------------------------------------------------------

# Fill missing numerical values with the median.
df["age"] = df["age"].fillna(df["age"].median())

# Fill missing categorical values with the mode.
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])


# ---------------------------------------------------------
# 5. VERIFY MISSING VALUES
# ---------------------------------------------------------

print("\nMissing values after preprocessing:")
print(df.isnull().sum())

print("\nDataset shape after removing unnecessary columns:")
print(df.shape)
# ---------------------------------------------------------
# 7. ENCODE CATEGORICAL FEATURES
# ---------------------------------------------------------

# Separate the target variable from the input features.
y = df["survived"]

# Remove the target from the feature dataframe.
X = df.drop(columns=["survived"])


# Convert Boolean columns to integers.
X["adult_male"] = X["adult_male"].astype(int)
X["alone"] = X["alone"].astype(int)


# Identify categorical columns.
categorical_columns = [
    "sex",
    "embarked",
    "class",
    "who",
    "embark_town"
]


# Apply one-hot encoding to categorical columns.
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded_array = encoder.fit_transform(X[categorical_columns])

encoded_columns = encoder.get_feature_names_out(categorical_columns)

encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoded_columns,
    index=X.index
)


# Remove original categorical columns.
X = X.drop(columns=categorical_columns)


# Combine numerical and encoded categorical features.
X = pd.concat([X, encoded_df], axis=1)


print("\n" + "=" * 60)
print("ENCODING COMPLETED")
print("=" * 60)

print("\nEncoded feature columns:")
print(X.columns.tolist())

print("\nFeature matrix shape after encoding:")
print(X.shape)

print("\nFirst 5 rows after encoding:")
print(X.head())
# ---------------------------------------------------------
# 8. SCALE NUMERICAL FEATURES
# ---------------------------------------------------------

# Select continuous numerical features for scaling.
numerical_columns = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

# Create a StandardScaler.
scaler = StandardScaler()

# Scale the selected numerical columns.
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])


print("\n" + "=" * 60)
print("SCALING COMPLETED")
print("=" * 60)

print("\nScaled numerical feature summary:")
print(X[numerical_columns].describe())

print("\nFirst 5 rows after scaling:")
print(X.head())
# ---------------------------------------------------------
# 9. EXPORT ML-READY FEATURES
# ---------------------------------------------------------

# Add the target variable back to the processed features.
ml_ready_data = X.copy()
ml_ready_data["survived"] = y


# Save the processed dataset as a CSV file.
OUTPUT_FILE = "W2D5_End_to_End_Preprocessing/titanic_processed.csv"

ml_ready_data.to_csv(OUTPUT_FILE, index=False)


print("\n" + "=" * 60)
print("EXPORT COMPLETED")
print("=" * 60)

print(f"\nML-ready dataset saved to: {OUTPUT_FILE}")
print(f"Final dataset shape: {ml_ready_data.shape}")

print("\nFinal dataset columns:")
print(ml_ready_data.columns.tolist())

print("\nTarget distribution:")
print(ml_ready_data["survived"].value_counts())

print("\nFirst 5 rows of ML-ready dataset:")
print(ml_ready_data.head())