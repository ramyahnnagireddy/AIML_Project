
"""Compare baseline and web-enhanced research outputs."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


def compare_reports(
    baseline: str,
    web_enhanced: str,
    source_count: int,
) -> str:
    """Create a transparent, evidence-based experiment comparison."""
    baseline_words = len(baseline.split())
    enhanced_words = len(web_enhanced.split())

    baseline_urls = baseline.count("http://") + baseline.count("https://")
    enhanced_urls = web_enhanced.count("http://") + web_enhanced.count(
        "https://"
    )

    if source_count:
        source_observation = (
            f"The search tool recorded {source_count} source(s). "
            "Inspect the saved source list to assess relevance and quality."
        )
    else:
        source_observation = (
    "No sources were recorded. The web-search experiment "
    "does not establish source-backed improvement."
)

    return f"""# W9D2 Research Quality Comparison

## Experiment setup

- Baseline: crew run without the web-search tool.
- Web-enhanced: crew run with the Wikipedia search tool available.
- Code-execution tool: restricted arithmetic calculator.
- Comparison method: word counts, URL counts, and captured source count.

## Observations

| Measure | Baseline | Web-enhanced |
|---|---:|---:|
| Word count | {baseline_words} | {enhanced_words} |
| URL occurrences | {baseline_urls} | {enhanced_urls} |
| Captured search sources | Not applicable | {source_count} |

## Source observation

{source_observation}

## Interpretation

The web-enhanced run can provide traceable search results when the API
succeeds and the researcher uses the tool. URL counts and word counts are
descriptive measurements, not measures of factual accuracy.

A larger report or more URLs does not by itself prove higher quality.
Review both reports for coverage, unsupported claims, relevance, balance,
and whether claims are supported by the captured sources.

## Limitations

- Wikipedia search results are not independently verified evidence.
- Search results can be incomplete, outdated, or irrelevant.
- Local model responses can vary between runs.
- The comparison does not establish causation or factual accuracy.
"""


def write_comparison(
    baseline: str,
    web_enhanced: str,
    source_count: int,
) -> Path:
    """Save the comparison report to the project's outputs folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "quality_comparison.md"
    destination.write_text(
        compare_reports(baseline, web_enhanced, source_count),
        encoding="utf-8",
    )
    return destination
