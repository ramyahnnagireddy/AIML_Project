
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    category: str
    route: str
    needs_review: bool
    human_decision: str
    response: str


def classify(state: AgentState) -> dict:
    """Classify an incoming request using simple, deterministic rules."""
    message = state["user_input"].lower()

    if any(word in message for word in ("urgent", "emergency", "critical")):
        category = "urgent"
    elif any(word in message for word in ("payment", "billing", "refund", "invoice")):
        category = "billing"
    elif any(word in message for word in ("code", "python", "error", "technical")):
        category = "technical"
    else:
        category = "general"

    return {"category": category}


def choose_next(
    state: AgentState,
) -> Literal["route", "respond"]:
    """Route urgent, billing, and technical requests through the routing node."""
    if state["category"] in {"urgent", "billing", "technical"}:
        return "route"
    return "respond"


def route(state: AgentState) -> dict:
    """Assign a handling route and flag urgent requests for human review."""
    category = state["category"]

    return {
        "route": f"{category}_support",
        "needs_review": category == "urgent",
    }


def respond(state: AgentState) -> dict:
    """Generate a response and pause for human approval when required."""
    if state.get("needs_review", False):
        decision = interrupt(
            {
                "message": "Human review required before responding.",
                "request": state["user_input"],
                "category": state["category"],
                "route": state["route"],
                "instructions": (
                    "Resume with {'decision': 'approve'} or "
                    "{'decision': 'reject'}."
                ),
            }
        )

        if decision.get("decision") == "approve":
            response = (
                "Your urgent request has been approved for priority handling."
            )
        else:
            response = (
                "Your request needs further review before priority handling."
            )
    else:
        category = state["category"]
        response = {
            "billing": "Your request has been directed to billing support.",
            "technical": "Your request has been directed to technical support.",
            "general": "Thanks for your message. How can I help you further?",
        }.get(category, "Your request has been received.")

    return {"response": response}


def build_graph():
    """Build the three-node graph with conditional routing and checkpoints."""
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify)
    builder.add_node("route", route)
    builder.add_node("respond", respond)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        choose_next,
        {
            "route": "route",
            "respond": "respond",
        },
    )
    builder.add_edge("route", "respond")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=MemorySaver())


def run_with_human_review(user_input: str) -> dict:
    """Run an urgent request until interrupted, then resume after review."""
    graph = build_graph()
    config = {"configurable": {"thread_id": "w10d2-human-review"}}

    initial = graph.invoke({"user_input": user_input}, config=config)

    if "__interrupt__" not in initial:
        return initial

    print("Human review interrupt:")
    print(initial["__interrupt__"])

    final = graph.invoke(
        Command(resume={"decision": "approve"}),
        config=config,
    )
    return final
