from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_home():
    """Test the health-check endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "ML Model Serving API is running"
    }


def test_predict():
    """Test the prediction endpoint."""
    response = client.post(
        "/predict",
        json={"features": [0, 0, 0, 0]},
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)
    