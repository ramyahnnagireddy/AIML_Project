
"""Automated tests for the production RAG API."""

import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "W8D1_Production_RAG_MLOps"

sys.path.insert(0, str(APP_DIR))

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "Production RAG API"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_metrics_endpoint():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "requests_total" in response.json()
    assert "server_errors_total" in response.json()


def test_query_endpoint_returns_rag_answer():
    fake_result = {
        "answer": "RAG combines retrieval with language generation.",
        "retrieved_chunks": [
            {"source": "rag_fundamentals.txt", "text": "Example", "score": 0.9},
            {"source": "rag_fundamentals.txt", "text": "Example", "score": 0.8},
            {"source": "machine_learning.txt", "text": "Example", "score": 0.7},
        ],
        "latency_seconds": 0.12,
    }

    with patch("app.main.rag_pipeline.ask_question", return_value=fake_result):
        response = client.post(
            "/query",
            json={"question": "What is RAG?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == fake_result["answer"]
    assert data["sources"] == [
        "machine_learning.txt",
        "rag_fundamentals.txt",
    ]
    assert data["latency_seconds"] == 0.12


def test_query_rejects_empty_question():
    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422


def test_query_rejects_missing_question():
    response = client.post("/query", json={})

    assert response.status_code == 422


def test_query_maps_pipeline_failure_to_502():
    with patch(
        "app.main.rag_pipeline.ask_question",
        side_effect=RuntimeError("Simulated pipeline failure"),
    ):
        response = client.post(
            "/query",
            json={"question": "What is RAG?"},
        )

    assert response.status_code == 502
    assert "RAG processing failed" in response.json()["detail"]

