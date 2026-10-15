
"""LangGraph workflow for local research question answering."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from research_assistant import ResearchAssistant


class ResearchState(TypedDict, total=False):
    """State passed between research workflow nodes."""

    question: str
    answer: str
    sources: list[dict]
    model: str


def build_research_graph(assistant: ResearchAssistant):
    """Build a small, testable retrieve-and-answer graph."""

    def research_node(state: ResearchState) -> ResearchState:
        result = assistant.answer(state["question"])
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "model": result["model"],
        }

    graph = StateGraph(ResearchState)
    graph.add_node("research", research_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", END)

    return graph.compile()
