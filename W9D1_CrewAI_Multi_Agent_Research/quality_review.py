"""Utilities for comparing baseline and web-enhanced research reports."""

import re


def assess_report(report: str) -> dict[str, int]:
    """Calculate simple, reproducible report-quality indicators."""
    words = re.findall(r"\b[\w'-]+\b", report)
    headings = re.findall(r"(?m)^\s*#{1,6}\s+.+$", report)
    urls = re.findall(r"https?://[^\s)\]>]+", report)

    return {
        "word_count": len(words),
        "heading_count": len(headings),
        "source_url_count": len(set(urls)),
    }


def build_comparison(
    baseline_report: str,
    web_report: str,
) -> str:
    """Create a Markdown comparison of the two report versions."""
    baseline = assess_report(baseline_report)
    enhanced = assess_report(web_report)

    def difference(metric: str) -> int:
        return enhanced[metric] - baseline[metric]

    baseline_words = baseline["word_count"]
    enhanced_words = enhanced["word_count"]

    if enhanced["source_url_count"] > baseline["source_url_count"]:
        source_finding = "The web-enhanced report contains more distinct source URLs."
    elif enhanced["source_url_count"] == baseline["source_url_count"]:
        source_finding = "Both reports contain the same number of distinct source URLs."
    else:
        source_finding = "The web-enhanced report contains fewer distinct source URLs."

    return f"""# W9D1 Research Quality Comparison
## Method

The same research topic was assigned to the same three-agent crew twice.
The baseline run did not provide a search tool to the Researcher.
The second run provided a Wikipedia web-search tool.

## Measured indicators

| Metric | Baseline | Web-enhanced | Difference |
|---|---:|---:|---:|
| Word count | {baseline_words} | {enhanced_words} | {difference("word_count"):+d} |
| Markdown headings | {baseline["heading_count"]} | {enhanced["heading_count"]} | {difference("heading_count"):+d} |
| Distinct source URLs | {baseline["source_url_count"]} | {enhanced["source_url_count"]} | {difference("source_url_count"):+d} |

## Observed findings

- {source_finding}
- Word-count change: {difference("word_count"):+d} words.
- Heading-count change: {difference("heading_count"):+d} headings.

## Limitations and interpretation
Source URL count measures whether reports contain traceable links; it does not prove that their claims are correct. Word count and heading count measure
report structure, not factual accuracy. Wikipedia search results should be
checked against their linked articles, and important claims should be verified
against authoritative primary sources.

The comparison describes the outputs of these two runs. Differences may also
reflect model variability, so they should not be interpreted as a controlled
proof that web search improves every aspect of quality.
"""


def write_comparison(
    baseline_report: str,
    web_report: str,
    output_path: str,
) -> str:
    """Write the comparison report and return its contents."""
    comparison = build_comparison(baseline_report, web_report)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(comparison)

    return comparison
