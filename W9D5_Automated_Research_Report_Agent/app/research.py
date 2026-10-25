
"""Utilities for producing auditable automated research reports."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"

_STOPWORDS = {
    "about", "after", "also", "and", "are", "benefit", "benefits",
    "between", "from", "have", "into", "limitations", "more", "risks",
    "that", "the", "their", "this", "through", "using", "what", "when",
    "where", "which", "with", "within", "without", "your", "applications",
    "application", "overview", "impact", "current", "future", "research",
    "analysis", "introduction", "explain", "explore", "discuss",
}


class _SnippetTextParser(HTMLParser):
    """Extract readable text from HTML snippets."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


def clean_snippet(snippet: str) -> str:
    """Remove HTML markup and decode entities from a search snippet."""
    parser = _SnippetTextParser()
    parser.feed(unescape(snippet))
    parser.close()
    return parser.text()


def _topic_terms(topic: str) -> set[str]:
    """Return meaningful topic terms for lightweight relevance filtering."""
    words = re.findall(r"[a-zA-Z0-9]+", topic.lower())
    return {
        word
        for word in words
        if len(word) >= 4 and word not in _STOPWORDS
    }


def _is_relevant(
    topic: str,
    title: str,
    snippet: str,
) -> bool:
    """Keep results sharing at least one meaningful topic term."""
    terms = _topic_terms(topic)
    if not terms:
        return True

    searchable_text = f"{title} {snippet}".lower()
    return any(term in searchable_text for term in terms)


def search_wikipedia(
    topic: str,
    limit: int = 5,
) -> list[dict[str, str]]:
    """Retrieve, clean, and relevance-filter Wikipedia search results."""
    if not topic.strip():
        raise ValueError("Research topic must not be empty.")
    if not 1 <= limit <= 10:
        raise ValueError("Search limit must be between 1 and 10.")

    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&srsearch={quote(topic)}"
        f"&srlimit={min(limit * 3, 30)}&format=json"
    )
    request = Request(
        url,
        headers={"User-Agent": "W9D5ResearchAgent/1.0"},
    )

    with urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    results = payload.get("query", {}).get("search", [])
    sources: list[dict[str, str]] = []

    for item in results:
        title = str(item.get("title", "")).strip()
        if not title:
            continue

        snippet = clean_snippet(str(item.get("snippet", "")))
        if not _is_relevant(topic, title, snippet):
            continue

        sources.append(
            {
                "title": title,
                "snippet": snippet,
                "url": (
                    "https://en.wikipedia.org/wiki/"
                    + quote(title.replace(" ", "_"))
                ),
            }
        )
        if len(sources) >= limit:
            break

    return sources


def format_sources(sources: list[dict[str, str]]) -> str:
    """Format retrieved results without claiming independent verification."""
    if not sources:
        return (
            "No relevant external search results were retrieved. "
            "External evidence is unavailable for this run."
        )

    sections = [
        "Retrieved search results (preliminary leads, not independently verified):"
    ]
    for source in sources:
        sections.append(
            f"- {source['title']}\n"
            f"  URL: {source['url']}\n"
            f"  Snippet: {source['snippet']}"
        )

    return "\n".join(sections)


def build_report(
    topic: str,
    research_notes: str,
    review_notes: str,
    sources: list[dict[str, str]],
) -> str:
    """Build a report with source and verification limitations disclosed."""
    if not topic.strip():
        raise ValueError("Research topic must not be empty.")

    generated_at = datetime.now(UTC).isoformat()
    source_section = format_sources(sources)

    return f"""# Automated Research Report: {topic}

- Generated (UTC): {generated_at}
- Pipeline: CrewAI + LangGraph
- Relevant external search results retrieved: {len(sources)}

## Executive Summary

{research_notes.strip()}

## Research Sources

{source_section}

## Quality Review

{review_notes.strip()}

## Limitations

- Search-result snippets are preliminary leads, not independently verified evidence.
- Relevance filtering is a simple keyword check and may miss useful sources or retain weak matches.
- Search snippets may omit important context from the full source.
- Claims should be checked against the linked pages and, where appropriate, independent research.
- Model-generated statements and review comments may contain errors or omissions.
- Human review is required before relying on or publishing this report.
"""


def save_report(content: str, output_dir: Path | None = None) -> Path:
    """Save the generated Markdown report."""
    destination = output_dir or OUTPUT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "automated_research_report.md"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def save_run_evidence(
    data: dict[str, Any],
    output_dir: Path | None = None,
) -> Path:
    """Save compact, machine-readable run evidence."""
    destination = output_dir or OUTPUT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "run_evidence.json"
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
