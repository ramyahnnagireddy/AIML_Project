
"""Tests for retrieval and the research assistant API."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Make the app modules importable when pytest runs from the repository root.
APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from main import app
from research_assistant import ResearchAssistant

client = TestClient(app)


def test_research_document_loads():
    """The default research notes should contain usable passages."""
    assistant = ResearchAssistant()

    assert assistant.chunks
    assert any("Retrieval-Augmented Generation" in chunk for chunk in assistant.chunks)


def test_retrieval_returns_relevant_evidence():
    """A RAG question should retrieve evidence about document retrieval."""
    assistant = ResearchAssistant()

    results = assistant.retrieve("How does retrieval augmented generation work?")

    assert results
    assert all("passage" in result for result in results)
    assert all(0 <= result["score"] <= 1 for result in results)


def test_empty_question_is_rejected_by_retriever():
    """Direct retrieval calls should reject blank questions."""
    assistant = ResearchAssistant()

    try:
        assistant.retrieve("   ")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected an empty question to be rejected")


def test_health_endpoint():
    """The health endpoint should report a healthy API."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_research_endpoint_returns_answer_and_sources():
    """The API should return an answer with supporting local evidence."""
    response = client.post(
        "/research",
        json={"question": "What is RAG?", "top_k": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["sources"]
    assert len(payload["sources"]) <= 2


def test_research_endpoint_rejects_blank_question():
    """Pydantic validation should reject a blank question."""
    response = client.post("/research", json={"question": "   "})

    assert response.status_code == 422
    