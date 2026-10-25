
"""Run the W9D1 CrewAI multi-agent research experiment."""

import re
from pathlib import Path
from typing import Any

from crewai import LLM, Agent, Crew, Process, Task
from quality_review import write_comparison
from web_search import (
    WikipediaSearchTool,
    format_sources_markdown,
    get_retrieved_sources,
    reset_search_log,
)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
TOPIC = (
    "Applications of artificial intelligence in healthcare: "
    "benefits, limitations, risks, and responsible deployment"
)

URL_PATTERN = re.compile(r"https?://[^\s<>\])]+", re.IGNORECASE)
REFERENCE_SECTION_PATTERN = re.compile(
    r"(?im)^#{1,6}\s*(?:references|sources|bibliography|works cited)"
    r"\b[^\n]*\n[\s\S]*$"
)

LOCAL_LLM = LLM(
    model="ollama_chat/qwen2.5:3b",
    base_url="http://localhost:11434",
    timeout=180,
)


def save_checkpoint(label: str, task_name: str, output: Any) -> None:
    """Save a completed task immediately, before the next task starts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = getattr(output, "raw", str(output))
    path = OUTPUT_DIR / f"{label}_{task_name}_checkpoint.md"
    path.write_text(str(raw), encoding="utf-8")
    print(f"\nCheckpoint saved: {path.name}", flush=True)


def clean_generated_text(text: str, allowed_urls: set[str]) -> str:
    """Remove unreturned URLs and generated reference-list claims."""
    text = REFERENCE_SECTION_PATTERN.sub("", text)
    text = re.sub(
        r"\[URL\]",
        "[source URL unavailable]",
        text,
        flags=re.IGNORECASE,
    )

    def keep_only_returned_urls(match: re.Match[str]) -> str:
        candidate = match.group(0).rstrip(".,;:")
        if candidate in allowed_urls:
            return candidate
        return "[unverified URL removed]"

    return URL_PATTERN.sub(keep_only_returned_urls, text).strip()


def source_limitation_note(use_web_search: bool) -> str:
    """Describe the search configuration without overstating evidence."""
    if use_web_search:
        return (
            "\n\n## Source-traceability note\n"
            "The source list records URLs returned by the configured "
            "Wikipedia search tool. Their inclusion does not independently "
            "verify every claim or establish that every source is authoritative."
        )

    return (
        "\n\n## Source limitations\n"
        "Live web search was disabled for this baseline run. Claims have "
        "not been verified against live sources, and no live source URLs "
        "are claimed."
    )


def build_crew(use_web_search: bool, label: str):
    """Configure three agents and save each task's completed output."""
    search_tool = WikipediaSearchTool()
    researcher_tools = [search_tool] if use_web_search else []

    researcher = Agent(
        role="AI Research Analyst",
        goal=(
            "Gather accurate, relevant information about {topic}, "
            "distinguishing established facts from uncertainty."
        ),
        backstory=(
            "Never invent sources. Use the search tool when available "
            "and report only titles and URLs it actually returns."
        ),
        llm=LOCAL_LLM,
        tools=researcher_tools,
        verbose=False,
        allow_delegation=False,
        max_iter=2,
    )

    writer = Agent(
        role="Technical Research Writer",
        goal="Write a concise, balanced report about {topic}.",
        backstory=(
            "Write clear Markdown. Do not invent references, URLs, "
            "statistics, or claims of source verification."
        ),
        llm=LOCAL_LLM,
        verbose=False,
        allow_delegation=False,
        max_iter=2,
    )

    reviewer = Agent(
        role="Research Quality Reviewer",
        goal="Critically review the report about {topic}.",
        backstory=(
            "Check coverage, clarity, balance, evidence limitations, "
            "ethical risks, and source traceability. Never invent facts."
        ),
        llm=LOCAL_LLM,
        verbose=False,
        allow_delegation=False,
        max_iter=2,
    )

    research_task = Task(
        description=(
            "Research {topic}. Provide concise notes on applications, "
            "benefits, limitations, privacy, safety, bias, and responsible "
            "deployment. Use actual search results when available. Never "
            "invent studies, titles, statistics, or URLs. If search is "
            "unavailable or returns no useful sources, state that clearly."
        ),
        expected_output=(
            "Concise structured research notes, including evidence "
            "limitations and actual source URLs when available."
        ),
        agent=researcher,
        callback=lambda output: save_checkpoint(
            label, "research", output
        ),
    )

    writing_task = Task(
        description=(
            "Use the research notes to write a concise Markdown report "
            "about {topic}. Include an introduction, key applications, "
            "benefits, limitations and risks, responsible deployment "
            "recommendations, and a conclusion. Do not invent citations, "
            "statistics, article titles, or URLs. Explicitly disclose "
            "missing source evidence."
        ),
        expected_output=(
            "A balanced Markdown report with clear headings and "
            "honest source limitations."
        ),
        agent=writer,
        context=[research_task],
        callback=lambda output: save_checkpoint(
            label, "report", output
        ),
    )

    review_task = Task(
        description=(
            "Review the report about {topic}. Return concise strengths, "
            "weaknesses, recommended corrections, and an overall quality "
            "assessment. Check clarity, coverage, unsupported claims, "
            "ethical concerns, and source traceability. Do not claim "
            "verification without evidence and do not invent citations."
        ),
        expected_output=(
            "A concise review with strengths, weaknesses, corrections, "
            "and explicit evidence limitations."
        ),
        agent=reviewer,
        context=[research_task, writing_task],
        callback=lambda output: save_checkpoint(
            label, "review", output
        ),
    )

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=False,
    )

    return crew, research_task, writing_task, review_task


def task_text(task: Task) -> str:
    """Retrieve a completed task's raw output."""
    output = task.output
    if output is None:
        return "No task output was recorded."
    return str(output.raw)


def run_experiment(use_web_search: bool) -> dict[str, str]:
    """Run one experiment and save filtered reports."""
    label = "web_enhanced" if use_web_search else "baseline"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 65}", flush=True)
    print(f"Starting {label.upper()} crew", flush=True)
    print(f"Topic: {TOPIC}", flush=True)
    print(f"Web search enabled: {use_web_search}", flush=True)
    print("Allow time for local model inference.", flush=True)
    print(f"{'=' * 65}", flush=True)

    crew, research_task, writing_task, review_task = build_crew(
        use_web_search=use_web_search,
        label=label,
    )

    crew.kickoff(inputs={"topic": TOPIC})

    research = task_text(research_task)
    report = task_text(writing_task)
    review = task_text(review_task)

    sources = get_retrieved_sources() if use_web_search else []
    allowed_urls = {source["url"] for source in sources}

    research = clean_generated_text(research, allowed_urls)
    report = clean_generated_text(report, allowed_urls)
    review = clean_generated_text(review, allowed_urls)

    note = source_limitation_note(use_web_search)
    research += note
    report += note

    if use_web_search:
        appendix = format_sources_markdown(sources)
        research += "\n\n" + appendix
        report += "\n\n" + appendix

    files = {
        f"{label}_research_notes.md": research,
        f"{label}_report.md": report,
        f"{label}_review.md": review,
    }

    for filename, content in files.items():
        (OUTPUT_DIR / filename).write_text(content, encoding="utf-8")
        print(f"Saved: outputs/{filename}", flush=True)

    print(f"Captured search results: {len(sources)}", flush=True)
    print(f"Completed {label} crew.", flush=True)

    return {"research": research, "report": report, "review": review}


def main() -> None:
    """Run baseline and enhanced experiments independently."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = run_experiment(use_web_search=False)

    # Clear the source log so it contains only enhanced-run results.
    reset_search_log()

    enhanced = run_experiment(use_web_search=True)

    comparison = write_comparison(
        baseline_report=baseline["report"],
        web_report=enhanced["report"],
        output_path=str(OUTPUT_DIR / "quality_comparison.md"),
    )

    print("\n" + comparison, flush=True)
    print("\nW9D1 experiment completed.", flush=True)


if __name__ == "__main__":
    main()
    
