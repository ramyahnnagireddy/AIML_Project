
"""Tests for the W10D1 stateful LangGraph."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import build_graph, choose_route, run_request


@pytest.mark.parametrize(
    ("user_input", "expected", "decision"),
    [
        ("What is LangGraph?", "general", "approve"),
        ("Explain Python lists.", "general", "approve"),
        (
            "urgent: delete customer records",
            "needs_review",
            "approve",
        ),
        (
            "How does MLflow track experiments?",
            "general",
            "approve",
        ),
        ("Should I share my password?", "needs_review", "reject"),
    ],
)
def test_five_scenarios(user_input, expected, decision):
    """Verify classification and response for five inputs."""
    app = build_graph(checkpointer=InMemorySaver())

    result = run_request(
        app=app,
        user_input=user_input,
        thread_id=f"test-{expected}-{user_input[:12]}",
        human_decision=decision,
    )

    assert result["classification"] == expected
    assert result["response"]
    if expected == "needs_review":
        assert result["human_decision"] == decision


def test_conditional_routing():
    """Verify each classification selects the correct next node."""
    assert choose_route(
        {"needs_review": True}
    ) == "review"
    assert choose_route(
        {"needs_review": False}
    ) == "respond"


def test_interrupt_pauses_and_resumes():
    """Verify a review request pauses and resumes with human input."""
    app = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "interrupt-resume-test"}}

    paused = app.invoke(
        {"user_input": "urgent: delete customer records"},
        config=config,
    )

    assert "__interrupt__" in paused
    assert paused["classification"] == "needs_review"

    resumed = app.invoke(
        Command(resume="approve"),
        config=config,
    )

    assert resumed["human_decision"] == "approve"
    assert "approved" in resumed["response"].lower()

