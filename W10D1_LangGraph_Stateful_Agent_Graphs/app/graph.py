
"""Three-node LangGraph with conditional routing and human review."""

from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """State passed between the graph nodes."""

    user_input: str
    classification: str
    needs_review: bool
    human_decision: str
    response: str


def classify_node(state: AgentState) -> dict:
    """Classify the request and flag sensitive actions for review."""
    message = state["user_input"].lower()

    review_keywords = (
        "urgent",
        "delete customer records",
        "share my password",
        "transfer money",
        "medical emergency",
    )
    needs_review = any(
        keyword in message for keyword in review_keywords
    )

    classification = "needs_review" if needs_review else "general"

    return {
        "classification": classification,
        "needs_review": needs_review,
    }


def choose_route(
    state: AgentState,
) -> Literal["review", "respond"]:
    """Choose the next node from the classification result."""
    if state["needs_review"]:
        return "review"
    return "respond"


def route_node(state: AgentState) -> dict:
    """Pause execution and wait for a human decision."""
    decision = interrupt(
        {
            "message": "Human review required before responding.",
            "user_input": state["user_input"],
            "classification": state["classification"],
            "choices": ["approve", "reject"],
        }
    )

    return {"human_decision": str(decision)}


def respond_node(state: AgentState) -> dict:
    """Generate a deterministic response from the current state."""
    if state["needs_review"]:
        decision = state.get("human_decision", "not provided")

        if decision.lower() == "approve":
            response = (
                "Human review approved. Proceed only with the "
                "authorized next steps."
            )
        else:
            response = (
                f"Human review decision: {decision}. "
                "Do not proceed without appropriate authorization."
            )
    else:
        response = (
            "General request received: "
            f"{state['user_input']}"
        )

    return {"response": response}


def build_graph(checkpointer=None):
    """Build and compile the three-node stateful graph."""
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify_node)
    builder.add_node("route", route_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "classify")

    builder.add_conditional_edges(
        "classify",
        choose_route,
        {
            "review": "route",
            "respond": "respond",
        },
    )

    builder.add_edge("route", "respond")
    builder.add_edge("respond", END)

    if checkpointer is None:
        checkpointer = InMemorySaver()

    return builder.compile(checkpointer=checkpointer)


def run_request(
    app,
    user_input: str,
    thread_id: str,
    human_decision: str = "approve",
) -> dict:
    """Run a request and resume it if human review interrupts it."""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"user_input": user_input}, config=config)

    if "__interrupt__" in result:
        result = app.invoke(
            Command(resume=human_decision),
            config=config,
        )

    return result
