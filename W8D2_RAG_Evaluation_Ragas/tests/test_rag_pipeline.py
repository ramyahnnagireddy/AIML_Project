
"""Unit tests for the W8D2 RAG pipeline."""

import pytest

from W8D2_RAG_Evaluation_Ragas.rag_pipeline import RAGPipeline, split_text


def test_split_text_returns_empty_list_for_empty_input() -> None:
    assert split_text("", chunk_size=100, chunk_overlap=10) == []
    assert split_text("   \n  ", chunk_size=100, chunk_overlap=10) == []


def test_split_text_creates_multiple_chunks() -> None:
    text = "RAG retrieves relevant information. " * 30
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_split_text_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        split_text("Sample text.", chunk_size=0, chunk_overlap=0)


def test_split_text_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_text("Sample text.", chunk_size=50, chunk_overlap=-1)


def test_split_text_rejects_overlap_equal_to_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_text("Sample text.", chunk_size=50, chunk_overlap=50)


def test_pipeline_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        RAGPipeline(top_k=0)


def test_pipeline_rejects_invalid_chunk_configuration() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        RAGPipeline(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap"):
        RAGPipeline(chunk_size=100, chunk_overlap=100)
