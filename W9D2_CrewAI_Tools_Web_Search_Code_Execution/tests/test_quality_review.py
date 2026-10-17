
"""Tests for research quality comparison."""

from quality_review import compare_reports


def test_comparison_includes_both_runs_and_source_count():
    result = compare_reports(
        baseline="Baseline research without live search.",
        web_enhanced="Research with https://en.wikipedia.org/",
        source_count=3,
    )

    assert "# W9D2 Research Quality Comparison" in result
    assert "Baseline" in result
    assert "Web-enhanced" in result
    assert "3 source(s)" in result


def test_comparison_discloses_missing_sources():
    result = compare_reports(
        baseline="Baseline report.",
        web_enhanced="Enhanced report without sources.",
        source_count=0,
    )

    assert "No sources were recorded" in result
    assert "does not establish source-backed improvement" in result


def test_comparison_does_not_claim_word_count_proves_quality():
    result = compare_reports(
        baseline="Short report.",
        web_enhanced="A longer report with extra words.",
        source_count=1,
    )

    assert "not measures of factual accuracy" in result
    assert "does not by itself prove higher quality" in result
    