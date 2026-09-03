"""
W4D4 - Train and save a machine learning model
for FastAPI model serving.
"""

from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train classification model
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X, y)

# Save trained model
model_path = BASE_DIR / "model.joblib"
joblib.dump(model, model_path)

print(f"Model trained successfully.")
print(f"Model saved to: {model_path}")
