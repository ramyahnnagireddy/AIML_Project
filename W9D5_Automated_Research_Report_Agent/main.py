
"""Run the W9D5 automated research report agent."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from urllib.error import URLError

from app.research import (
    build_report,
    save_report,
    save_run_evidence,
    search_wikipedia,
)
from app.tracking import evaluate_with_ragas, log_research_run
from app.workflow import ResearchState, build_workflow
from crewai import LLM, Agent, Crew, Process, Task

MODEL_NAME = "ollama/qwen2.5:3b"


def make_llm() -> LLM:
    """Configure CrewAI to use the existing local Ollama model."""
    return LLM(model=MODEL_NAME, base_url="http://localhost:11434")


def run_research_crew(topic: str, source_text: str) -> str:
    """Use a CrewAI research agent to produce a concise research brief."""
    researcher = Agent(
        role="Research Analyst",
        goal=(
            "Summarize relevant findings and identify evidence gaps "
            "without inventing facts, citations, or statistics."
        ),
        backstory=(
            "You are a cautious analyst. Distinguish supplied search "
            "evidence from general knowledge and disclose uncertainty."
        ),
        llm=make_llm(),
        allow_delegation=False,
        verbose=False,
    )

    research_task = Task(
        description=(
            f"Research topic: {topic}\n\n"
            f"Observed Wikipedia search results:\n{source_text}\n\n"
            "Produce a concise research brief covering key concepts, "
            "applications, benefits, risks, limitations, and open questions. "
            "Do not invent citations, statistics, or source contents. "
            "Treat search snippets as leads, not verified evidence."
        ),
        expected_output=(
            "A structured research brief with findings and explicit "
            "evidence limitations."
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=False,
    )
    return str(crew.kickoff())


def run_review_crew(topic: str, draft: str) -> str:
    """Use a separate CrewAI agent to critique the generated report draft."""
    reviewer = Agent(
        role="Research Quality Reviewer",
        goal=(
            "Critically review a research report for unsupported claims, "
            "missing perspectives, clarity, and balanced coverage."
        ),
        backstory=(
            "You are a careful reviewer. You do not assume claims are true "
            "merely because they appear in a model-generated report."
        ),
        llm=make_llm(),
        allow_delegation=False,
        verbose=False,
    )

    review_task = Task(
        description=(
            f"Research topic: {topic}\n\n"
            "Review the following report draft:\n\n"
            f"{draft}\n\n"
            "Identify unsupported or overconfident statements, missing "
            "benefits and risks, unclear wording, and important limitations. "
            "Give concrete recommendations. Do not invent facts or claim "
            "that you independently verified the sources."
        ),
        expected_output=(
            "A structured quality review describing strengths, issues, "
            "recommended improvements, and remaining limitations."
        ),
        agent=reviewer,
    )

    crew = Crew(
        agents=[reviewer],
        tasks=[review_task],
        process=Process.sequential,
        verbose=False,
    )
    return str(crew.kickoff())


def run_research(topic: str) -> ResearchState:
    """Execute retrieval, research, report drafting, and review via LangGraph."""
    if not topic.strip():
        raise ValueError("Research topic must not be empty.")

    sources: list[dict[str, str]] = []
    errors: list[str] = []

    try:
        sources = search_wikipedia(topic)
    except (URLError, OSError, TimeoutError, ValueError) as exc:
        errors.append(
            f"External search unavailable: {type(exc).__name__}: {exc}"
        )

    source_text = "\n".join(
        f"- {item['title']}: {item['snippet']} ({item['url']})"
        for item in sources
    )
    if not source_text:
        source_text = "No external search results were available."

    def research_node(state: ResearchState) -> ResearchState:
        research_notes = run_research_crew(
            state["topic"],
            source_text,
        )
        return {
            "research_notes": research_notes,
            "sources": sources,
            "errors": errors,
        }

    def writing_node(state: ResearchState) -> ResearchState:
        draft = build_report(
            topic=state["topic"],
            research_notes=state["research_notes"],
            review_notes="Quality review is pending.",
            sources=state.get("sources", []),
        )
        return {"report": draft}

    def review_node(state: ResearchState) -> ResearchState:
        review_notes = run_review_crew(
            state["topic"],
            state["report"],
        )
        final_report = build_report(
            topic=state["topic"],
            research_notes=state["research_notes"],
            review_notes=review_notes,
            sources=state.get("sources", []),
        )
        return {
            "review_notes": review_notes,
            "report": final_report,
        }

    workflow = build_workflow(
        research_node,
        writing_node,
        review_node,
    )
    return workflow.invoke(
        {
            "topic": topic,
            "sources": sources,
            "errors": errors,
        }
    )


def main() -> None:
    """Run the agent and save its report, evaluation inputs, and evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default=(
            "Applications, benefits, risks, and limitations of AI in healthcare"
        ),
        help="Research topic for the report.",
    )
    args = parser.parse_args()

    state = run_research(args.topic)
    report_path = save_report(state["report"])

    evaluation = evaluate_with_ragas(
        question=args.topic,
        answer=state["report"],
        contexts=[
            f"{source['title']}: {source['snippet']}"
            for source in state.get("sources", [])
        ],
    )

    run_id = log_research_run(
        topic=args.topic,
        source_count=len(state.get("sources", [])),
        report_length=len(state["report"]),
        status="completed",
    )

    evidence = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "topic": args.topic,
        "model": MODEL_NAME,
        "source_count": len(state.get("sources", [])),
        "report_path": str(report_path),
        "mlflow_run_id": run_id,
        "evaluation": evaluation,
        "errors": state.get("errors", []),
    }
    evidence_path = save_run_evidence(evidence)

    print(f"Report saved: {report_path}")
    print(f"Evidence saved: {evidence_path}")
    print(f"Sources retrieved: {len(state.get('sources', []))}")
    print(f"MLflow run: {run_id or 'not recorded'}")
    print(f"Evaluation status: {evaluation['evaluation_status']}")

    if state.get("errors"):
        print("Warnings:")
        for error in state["errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
