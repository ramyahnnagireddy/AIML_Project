"""Unit tests for the W7D5 multi-document RAG pipeline."""

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import rag_pipeline
from evaluate_rag import calculate_scores


def test_load_documents_returns_text_chunks():
    documents = rag_pipeline.load_documents()

    assert documents
    assert all(item["source"] for item in documents)
    assert all(item["text"].strip() for item in documents)


def test_retrieve_chunks_returns_relevant_results():
    documents = rag_pipeline.load_documents()

    results = rag_pipeline.retrieve_chunks(
        "What is Retrieval-Augmented Generation?",
        documents,
    )

    assert results
    assert len(results) <= rag_pipeline.TOP_K
    assert all(item["score"] > 0 for item in results)


def test_retrieve_chunks_empty_question():
    documents = rag_pipeline.load_documents()

    assert rag_pipeline.retrieve_chunks("", documents) == []


def test_calculate_scores_returns_values_between_zero_and_one():
    support, relevance = calculate_scores(
        "What are text embeddings?",
        "Text embeddings convert text into numerical vectors.",
        ["Text embedding models convert text into numerical vectors."],
    )

    assert 0.0 <= support <= 1.0
    assert 0.0 <= relevance <= 1.0
    assert support > 0
    assert relevance > 0


def test_evaluation_evidence_exists():
    output_file = rag_pipeline.OUTPUTS_DIR / "ragas_evaluation.json"

    assert output_file.exists()

    data = json.loads(output_file.read_text(encoding="utf-8"))

    assert data["sample_count"] >= 1
    assert len(data["scores"]) == data["sample_count"]
    assert data["evaluation_method"] == "deterministic lexical indicators"
    