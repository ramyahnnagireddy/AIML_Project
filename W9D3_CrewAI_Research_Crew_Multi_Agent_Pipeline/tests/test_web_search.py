
"""Tests for healthcare-focused Wikipedia search."""

from __future__ import annotations

from unittest.mock import Mock, patch

from W9D3_CrewAI_Research_Crew_Multi_Agent_Pipeline.web_search import (
    _clean_text,
    _is_relevant,
    search_web,
)


def test_clean_text_removes_html_and_decodes_entities() -> None:
    result = _clean_text(
        '<span class="searchmatch">Medical</span> &amp; healthcare'
    )

    assert result == "Medical & healthcare"


def test_is_relevant_accepts_healthcare_results() -> None:
    assert _is_relevant(
        "Clinical decision support system",
        "AI assists medical professionals with patient care.",
    )


def test_is_relevant_rejects_unrelated_content() -> None:
    assert not _is_relevant(
        "Generative AI pornography",
        "AI-generated sexual content.",
    )


@patch(
    "W9D3_CrewAI_Research_Crew_Multi_Agent_Pipeline"
    ".web_search.requests.Session"
)
def test_search_web_returns_ranked_unique_relevant_sources(
    mock_session_class: Mock,
) -> None:
    session = mock_session_class.return_value
    response = Mock()
    response.json.return_value = {
        "query": {
            "search": [
                {
                    "title": "Artificial intelligence in healthcare",
                    "snippet": (
                        "AI supports medical diagnosis and patient monitoring."
                    ),
                },
                {
                    "title": "Generative AI pornography",
                    "snippet": "AI-generated sexual content.",
                },
                {
                    "title": "Clinical decision support system",
                    "snippet": (
                        "A system used in medicine for clinical decisions."
                    ),
                },
            ]
        }
    }
    session.get.return_value = response

    results = search_web("AI in healthcare", max_results=5)

    assert results
    assert len(results) == 2
    assert all("title" in item for item in results)
    assert all("snippet" in item for item in results)
    assert all("url" in item for item in results)
    assert all(
        "pornography" not in item["title"].casefold()
        for item in results
    )
    assert results[0]["title"] == "Artificial intelligence in healthcare"
    assert len({item["url"] for item in results}) == len(results)


def test_search_web_returns_empty_for_nonpositive_limit() -> None:
    assert search_web("AI in healthcare", max_results=0) == []
    