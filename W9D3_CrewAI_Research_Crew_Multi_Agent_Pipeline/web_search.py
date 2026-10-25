
"""Healthcare-focused Wikipedia search for the W9D3 research crew."""

from __future__ import annotations

import html
import re
from typing import Any

import requests

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

SEARCH_QUERIES = (
    "artificial intelligence in healthcare",
    "artificial intelligence medical diagnosis clinical decision support",
    "artificial intelligence medical imaging drug discovery",
)

HEALTHCARE_TERMS = (
    "healthcare",
    "health care",
    "medical",
    "medicine",
    "clinical",
    "patient",
    "hospital",
    "diagnosis",
    "diagnostic",
    "radiology",
    "medical imaging",
    "drug discovery",
    "surgery",
    "pathology",
    "biomedical",
    "health data",
)

EXCLUDED_TERMS = (
    "pornography",
    "pornographic",
    "sexual content",
)


def _clean_text(value: str) -> str:
    """Remove HTML markup and decode HTML entities."""
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def _is_relevant(title: str, snippet: str) -> bool:
    """Check whether a search result is relevant to healthcare."""
    combined = f"{title} {snippet}".casefold()

    if any(term in combined for term in EXCLUDED_TERMS):
        return False

    return any(term in combined for term in HEALTHCARE_TERMS)


def _relevance_score(title: str, snippet: str) -> int:
    """Give healthcare-specific titles a higher ranking."""
    title_lower = title.casefold()
    snippet_lower = snippet.casefold()

    score = sum(
        3 for term in HEALTHCARE_TERMS if term in title_lower
    )
    score += sum(
        1 for term in HEALTHCARE_TERMS if term in snippet_lower
    )
    return score


def search_web(topic: str, max_results: int = 5) -> list[dict[str, str]]:
    """Retrieve relevant healthcare sources from Wikipedia's search API."""
    if max_results < 1:
        return []

    session = requests.Session()
    results_by_url: dict[str, dict[str, Any]] = {}

    try:
        for query in SEARCH_QUERIES:
            response = session.get(
                WIKIPEDIA_API_URL,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": f"{topic} {query}",
                    "format": "json",
                    "srlimit": 10,
                },
                headers={"User-Agent": "W9D3ResearchCrew/1.0"},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()

            search_results = (
                payload.get("query", {}).get("search", [])
            )

            for item in search_results:
                title = _clean_text(str(item.get("title", "")))
                snippet = _clean_text(str(item.get("snippet", "")))

                if not title or not _is_relevant(title, snippet):
                    continue

                page_title = title.replace(" ", "_")
                url = (
                    "https://en.wikipedia.org/wiki/"
                    + requests.utils.quote(page_title, safe="()_-'")
                )

                candidate = {
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                    "score": _relevance_score(title, snippet),
                }

                previous = results_by_url.get(url)
                if (
                    previous is None
                    or candidate["score"] > previous["score"]
                ):
                    results_by_url[url] = candidate

    finally:
        session.close()

    ranked_results = sorted(
        results_by_url.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    return [
        {
            "title": item["title"],
            "snippet": item["snippet"],
            "url": item["url"],
        }
        for item in ranked_results[:max_results]
    ]
