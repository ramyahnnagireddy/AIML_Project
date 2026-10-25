
"""Tests for report creation, source cleanup, and workflow behavior."""

import json

import pytest
from app.research import (
    _is_relevant,
    build_report,
    clean_snippet,
    save_report,
    save_run_evidence,
)
from app.tracking import evaluate_with_ragas
from app.workflow import build_workflow


def test_report_discloses_missing_sources():
    report = build_report(
        topic="AI in education",
        research_notes="AI can support learning when used responsibly.",
        review_notes="Verify factual claims before publication.",
        sources=[],
    )

    assert "# Automated Research Report: AI in education" in report
    assert "No relevant external search results were retrieved" in report
    assert "Human review is required" in report


def test_report_includes_retrieved_source():
    report = build_report(
        topic="RAG",
        research_notes="Research notes.",
        review_notes="Review notes.",
        sources=[
            {
                "title": "Retrieval-augmented generation",
                "snippet": "A search-result snippet.",
                "url": (
                    "https://en.wikipedia.org/wiki/"
                    "Retrieval-augmented_generation"
                ),
            }
        ],
    )

    assert "Retrieval-augmented generation" in report
    assert "https://en.wikipedia.org/" in report
    assert "not independently verified" in report


def test_clean_snippet_removes_html_and_decodes_entities():
    cleaned = clean_snippet(
        "AI <span class=\"searchmatch\">in</span> "
        "&quot;healthcare&quot;"
    )

    assert cleaned == 'AI in "healthcare"'
    assert "<span" not in cleaned
    assert "&quot;" not in cleaned


def test_relevance_filter_keeps_topic_match():
    assert _is_relevant(
        "AI in healthcare",
        "Artificial intelligence in healthcare",
        "AI is used with healthcare data.",
    )


def test_relevance_filter_rejects_unrelated_result():
    assert not _is_relevant(
        "AI in healthcare",
        "Regulation of artificial intelligence",
        "AI regulation and risks across many applications.",
    )


def test_save_report_and_evidence(tmp_path):
    report_path = save_report("# Test report", tmp_path)
    evidence_path = save_run_evidence({"status": "completed"}, tmp_path)

    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8") == "# Test report\n"
    assert json.loads(evidence_path.read_text(encoding="utf-8")) == {
        "status": "completed"
    }


def test_evaluation_without_context_does_not_invent_scores():
    result = evaluate_with_ragas(
        question="What is RAG?",
        answer="RAG retrieves relevant context.",
        contexts=[],
    )

    assert result["evaluation_status"] == "not_evaluated_no_contexts"
    assert result["scores"] == {}
    assert "not run" in result["note"]


def test_evaluation_rejects_empty_question():
    with pytest.raises(ValueError):
        evaluate_with_ragas("", "An answer.", ["Context"])


def test_evaluation_rejects_empty_answer():
    with pytest.raises(ValueError):
        evaluate_with_ragas("A question?", "", ["Context"])


def test_langgraph_workflow():
    def research(state):
        return {"research_notes": "Findings", "sources": []}

    def write(state):
        return {"report": state["research_notes"] + " report"}

    def review(state):
        return {"review_notes": "Looks clear"}

    workflow = build_workflow(research, write, review)
    result = workflow.invoke({"topic": "Testing"})

    assert result["research_notes"] == "Findings"
    assert result["report"] == "Findings report"
    assert result["review_notes"] == "Looks clear"
    