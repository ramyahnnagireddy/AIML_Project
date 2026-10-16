
"""W9D3: Three-agent CrewAI research pipeline using local Ollama."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from crewai import LLM, Agent, Crew, Process, Task
from web_search import search_web

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
TOPIC = "Applications, benefits, risks, and limitations of AI in healthcare"

LOCAL_LLM = LLM(
    model="ollama/qwen2.5:3b",
    base_url="http://localhost:11434",
)


def create_agents() -> tuple[Agent, Agent, Agent]:
    """Create the researcher, writer, and reviewer agents."""
    researcher = Agent(
        role="Research Analyst",
        goal=(
            "Identify relevant information, distinguish facts from uncertainty, "
            "and identify evidence gaps."
        ),
        backstory=(
            "You are a careful research analyst who values reliable evidence, "
            "balanced analysis, and clear acknowledgement of limitations."
        ),
        llm=LOCAL_LLM,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Technical Research Writer",
        goal=(
            "Produce a well-structured report based on the research findings "
            "and supplied source material."
        ),
        backstory=(
            "You are an experienced technical writer who explains complex "
            "subjects clearly and never invents citations or evidence."
        ),
        llm=LOCAL_LLM,
        verbose=True,
        allow_delegation=False,
    )

    reviewer = Agent(
        role="Research Quality Reviewer",
        goal=(
            "Evaluate relevance, clarity, balance, evidence quality, "
            "unsupported claims, and missing limitations."
        ),
        backstory=(
            "You are a critical reviewer who distinguishes sourced information "
            "from assumptions and gives practical improvement suggestions."
        ),
        llm=LOCAL_LLM,
        verbose=True,
        allow_delegation=False,
    )

    return researcher, writer, reviewer


def build_crew(topic: str, source_text: str | None = None) -> Crew:
    """Build a sequential research, writing, and review crew."""
    researcher, writer, reviewer = create_agents()

    if source_text:
        evidence = (
            "The following are retrieved search-result snippets. Treat them "
            "as leads, not verified full articles. Do not invent facts or "
            "sources.\n\n"
            f"{source_text}"
        )
    else:
        evidence = (
            "No external search results were supplied. Use your knowledge "
            "cautiously, avoid invented citations, and identify claims needing "
            "verification and limitations of this baseline."
        )

    research_task = Task(
        description=(
            f"Research this topic: {topic}\n\n"
            f"Research context:\n{evidence}\n\n"
            "Identify applications, potential benefits, risks, limitations, "
            "and unresolved questions. Distinguish general knowledge from "
            "information supported by the supplied search results. Do not "
            "claim a source proves something unless its supplied content "
            "supports that statement."
        ),
        expected_output=(
            "A concise research brief containing key findings, risks, "
            "limitations, and evidence gaps."
        ),
        agent=researcher,
    )

    writing_task = Task(
        description=(
            f"Write a balanced report about: {topic}\n"
            "Use the research brief from the previous task. Include an "
            "introduction, applications, benefits, risks, limitations, and "
            "conclusion. In the web-enhanced run, identify the supplied "
            "source URLs as search leads and disclose their limitations. "
            "Do not fabricate citations, statistics, or quotations."
        ),
        expected_output=(
            "A readable Markdown research report with clear sections, "
            "balanced analysis, and transparent evidence limitations."
        ),
        agent=writer,
        context=[research_task],
    )

    review_task = Task(
        description=(
            "Critically review the writer's draft. Assess relevance, "
            "organization, clarity, balance, evidence quality, unsupported "
            "claims, and coverage of limitations. Return the reviewed report "
            "followed by a section titled 'Reviewer Assessment' containing "
            "strengths, weaknesses, and specific recommendations. Do not "
            "claim external verification unless it actually occurred."
        ),
        expected_output=(
            "A reviewed Markdown report followed by a structured quality "
            "assessment with strengths, weaknesses, and recommendations."
        ),
        agent=reviewer,
        context=[research_task, writing_task],
    )

    return Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=True,
    )


def save_text(path: Path, content: str) -> None:
    """Write output using UTF-8."""
    path.write_text(content.strip() + "\n", encoding="utf-8")



def run_pipeline(web_only: bool = False) -> None:
    """Run baseline and enhanced research, or refresh only the enhanced run."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not web_only:
        print(f"Research topic: {TOPIC}")
        print("Run 1/2: baseline research")

        baseline_result = str(build_crew(TOPIC).kickoff())
        save_text(OUTPUT_DIR / "baseline_report.md", baseline_result)
    else:
        baseline_path = OUTPUT_DIR / "baseline_report.md"
        if not baseline_path.exists():
            raise FileNotFoundError(
                "baseline_report.md does not exist. Run the full pipeline first."
            )
        print(f"Keeping existing baseline report: {baseline_path}")

    print("Searching Wikipedia for healthcare-related source leads...")
    sources = search_web(TOPIC, max_results=5)
    source_text = json.dumps(sources, ensure_ascii=False, indent=2)

    (OUTPUT_DIR / "retrieved_sources.json").write_text(
        source_text + "\n",
        encoding="utf-8",
    )

    if not sources:
        raise RuntimeError(
            "No relevant web results were retrieved. Check internet access "
            "and search relevance before running the enhanced crew."
        )

    print(f"Retrieved {len(sources)} relevant source leads.")
    for source in sources:
        print(f"- {source['title']}: {source['url']}")

    print("Running web-enhanced research crew...")
    enhanced_result = str(build_crew(TOPIC, source_text).kickoff())
    save_text(OUTPUT_DIR / "web_enhanced_report.md", enhanced_result)

    baseline_path = OUTPUT_DIR / "baseline_report.md"
    comparison = f"""# W9D3 Research Comparison

- Topic: {TOPIC}
- Generated at (UTC): {datetime.now(UTC).isoformat()}
- Model: Ollama `qwen2.5:3b`
- Baseline report: `baseline_report.md`
- Web-enhanced report: `web_enhanced_report.md`
- Retrieved source leads: `retrieved_sources.json`
- Number of relevant search results: {len(sources)}

## Comparison method

Both reports use the same topic, local model, and three-agent sequential
workflow. The baseline uses no retrieved search results. The enhanced run
receives Wikipedia search-result titles, snippets, and URLs.

## Manual review checklist

Compare the saved reports for:

1. Relevance to AI in healthcare.
2. Coverage of applications, benefits, risks, and limitations.
3. Usefulness and specificity of the retrieved source leads.
4. Transparency about uncertainty and evidence gaps.
5. Unsupported claims or invented citations.
6. Clarity, organization, and practical recommendations.

## Evidence limitations

Search-result snippets are leads, not verified full articles. Their presence
does not establish that every claim in the enhanced report is accurate.
Review the source links and both reports before concluding whether quality
improved.

## Reviewer conclusion

Complete this section after inspecting both reports. Record concrete
differences and any remaining weaknesses. Do not claim improvement without
supporting evidence.
"""
    save_text(OUTPUT_DIR / "quality_comparison.md", comparison)

    print("\nResearch run completed.")
    print(f"Evidence directory: {OUTPUT_DIR}")
    print("Saved retrieved_sources.json")
    print("Saved web_enhanced_report.md")
    print("Saved quality_comparison.md")
    if not web_only:
        print("Saved baseline_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the W9D3 three-agent research pipeline."
    )
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="Refresh the web-enhanced report without rerunning the baseline.",
    )
    args = parser.parse_args()
    run_pipeline(web_only=args.web_only)
    