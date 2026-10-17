
"""LangGraph orchestration for the automated research report agent."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class ResearchState(TypedDict, total=False):
    """State shared by each node in the research workflow."""

    topic: str
    sources: list[dict[str, str]]
    research_notes: str
    review_notes: str
    report: str
    errors: list[str]


def build_workflow(
    research_node,
    writing_node,
    review_node,
):
    """Build a linear, testable research-to-report workflow."""
    graph = StateGraph(ResearchState)
    graph.add_node("research", research_node)
    graph.add_node("write_report", writing_node)
    graph.add_node("review", review_node)

    graph.add_edge(START, "research")
    graph.add_edge("research", "write_report")
    graph.add_edge("write_report", "review")
    graph.add_edge("review", END)

    return graph.compile()
