
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load the real Diabetes dataset
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

print("===== DATASET INFORMATION =====")
print("Features shape:", X.shape)
print("Target shape:", y.shape)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n===== TRAIN/TEST SPLIT =====")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Create the three regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0)
}


# Train, predict, and evaluate each model
results = []

for name, model in models.items():

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    })

    print(f"\n===== {name.upper()} =====")

    if name == "Linear Regression":
        print("Intercept:", model.intercept_)
        print("Coefficients:")

        for feature, coefficient in zip(
            diabetes.feature_names,
            model.coef_
        ):
            print(f"{feature}: {coefficient}")

    else:
        print("Intercept:", model.intercept_)


# Create a comparison table
results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

# Save comparison results
results_df.to_csv("regression_model_comparison.csv", index=False)


# Train Linear Regression again for plotting
linear_model = models["Linear Regression"]
linear_predictions = linear_model.predict(X_test)


# Predicted vs Actual plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, linear_predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Predicted vs Actual")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
plt.show()


# Calculate residuals
residuals = y_test - linear_predictions


# Residual plot
plt.figure(figsize=(8, 6))
plt.scatter(linear_predictions, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Linear Regression: Residual Plot")

plt.tight_layout()
plt.savefig("residuals.png")
plt.show()
