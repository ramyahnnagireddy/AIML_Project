
from app.graph import build_graph
from langgraph.types import Command

SAMPLE_CASES = [
    ("I have a Python error in my code.", "technical", "technical_support"),
    ("I need help with a payment refund.", "billing", "billing_support"),
    (
        "This is an urgent emergency and needs attention.",
        "urgent",
        "urgent_support",
    ),
    ("Can you explain how AI works?", "general", None),
    ("My invoice has an incorrect amount.", "billing", "billing_support"),
]


def test_five_inputs_are_classified_and_routed() -> None:
    graph = build_graph()

    for index, (user_input, expected_category, expected_route) in enumerate(
        SAMPLE_CASES,
        start=1,
    ):
        result = graph.invoke(
            {"user_input": user_input},
            config={
                "configurable": {
                    "thread_id": f"test-routing-{index}",
                }
            },
            interrupt_before=["respond"],
        )

        assert result["category"] == expected_category
        assert result.get("route") == expected_route


def test_general_request_receives_response() -> None:
    graph = build_graph()

    result = graph.invoke(
        {"user_input": "Can you explain how AI works?"},
        config={"configurable": {"thread_id": "test-general-response"}},
    )

    assert result["category"] == "general"
    assert "response" in result
    assert "help" in result["response"].lower()


def test_billing_request_receives_response() -> None:
    graph = build_graph()

    result = graph.invoke(
        {"user_input": "I need a refund for my payment."},
        config={"configurable": {"thread_id": "test-billing-response"}},
    )

    assert result["category"] == "billing"
    assert result["route"] == "billing_support"
    assert "billing support" in result["response"].lower()


def test_urgent_request_interrupts_and_resumes() -> None:
    graph = build_graph()
    config = {
        "configurable": {
            "thread_id": "test-human-review",
        }
    }

    paused = graph.invoke(
        {
            "user_input": "Urgent: a critical system error needs attention."
        },
        config=config,
    )

    assert "__interrupt__" in paused
    assert paused["__interrupt__"]
    assert paused["__interrupt__"][0].value["category"] == "urgent"

    resumed = graph.invoke(
        Command(resume={"decision": "approve"}),
        config=config,
    )

    assert resumed["category"] == "urgent"
    assert resumed["needs_review"] is True
    assert "approved" in resumed["response"].lower()


def test_urgent_request_can_be_rejected() -> None:
    graph = build_graph()
    config = {
        "configurable": {
            "thread_id": "test-human-rejection",
        }
    }

    paused = graph.invoke(
        {
            "user_input": "Emergency: critical service issue."
        },
        config=config,
    )

    assert "__interrupt__" in paused

    resumed = graph.invoke(
        Command(resume={"decision": "reject"}),
        config=config,
    )

    assert "further review" in resumed["response"].lower()
    