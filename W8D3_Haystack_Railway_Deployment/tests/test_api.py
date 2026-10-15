import sys
from pathlib import Path

from fastapi.testclient import TestClient

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from main import app


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["message"] == "Haystack Retrieval API"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert "bm25" in response.json()["initialized"]
        assert "dense" in response.json()["initialized"]


def test_dense_query():
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "question": "What is machine learning?",
                "retriever": "dense",
                "top_k": 3,
            },
        )
        assert response.status_code == 200
        assert len(response.json()["results"]) > 0


def test_bm25_query():
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "question": "What is MLOps?",
                "retriever": "bm25",
                "top_k": 3,
            },
        )
        assert response.status_code == 200
        assert len(response.json()["results"]) > 0


def test_invalid_retriever():
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "question": "What is machine learning?",
                "retriever": "invalid",
                "top_k": 3,
            },
        )
        assert response.status_code == 400
