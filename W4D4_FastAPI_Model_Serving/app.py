"""
W4D4 - FastAPI Model Serving Endpoint

Provides an API endpoint for making predictions
using a trained machine learning model.
"""

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


# Create FastAPI application
app = FastAPI(
    title="ML Model Serving API",
    description="FastAPI endpoint for Iris classification",
    version="1.0.0",
)


# Locate the saved model
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

# Load the trained model
model = joblib.load(MODEL_PATH)


# Define the prediction request structure
class PredictionRequest(BaseModel):
    features: list[float]


# Define the prediction response structure
class PredictionResponse(BaseModel):
    prediction: int


@app.get("/")
def home():
    """Health-check endpoint."""
    return {"message": "ML Model Serving API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Generate a prediction using the trained model."""

    # Convert input features into the format expected by the model
    input_data = np.array(request.features).reshape(1, -1)

    # Generate prediction
    prediction = model.predict(input_data)[0]

    return {"prediction": int(prediction)}

    