"""Offline tests for W9D1 research quality comparison."""

from W9D1_CrewAI_Multi_Agent_Research.quality_review import (
    assess_report,
    build_comparison,
)


def test_assess_report_counts_words_headings_and_urls():
    report = (
        "# Research\n\n"
        "AI supports medical research.\n\n"
        "## Sources\n"
        "https://example.com/article\n"
    )

    metrics = assess_report(report)

    assert metrics["word_count"] > 0
    assert metrics["heading_count"] == 2
    assert metrics["source_url_count"] == 1


def test_duplicate_urls_are_counted_once():
    report = (
        "First reference: https://example.com/article\n"
        "Repeated reference: https://example.com/article\n"
    )

    assert assess_report(report)["source_url_count"] == 1


def test_empty_report_has_zero_metrics():
    assert assess_report("") == {
        "word_count": 0,
        "heading_count": 0,
        "source_url_count": 0,
    }


def test_comparison_includes_both_versions():
    comparison = build_comparison(
        "# Baseline\nA short report without sources.",
        (
            "# Web Report\n\n"
            "## Evidence\n"
            "A longer report with evidence: https://example.com/source\n"
        ),
    )

    assert "# W9D1 Research Quality Comparison" in comparison
    assert "Baseline" in comparison
    assert "Web-enhanced" in comparison
    assert "source URLs" in comparison


def test_comparison_explains_limitations():
    comparison = build_comparison("Baseline text.", "Enhanced text.")

    assert "Limitations" in comparison
    assert "does not prove" in comparison
