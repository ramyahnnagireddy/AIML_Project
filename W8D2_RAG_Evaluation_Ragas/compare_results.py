
"""Compare W8D2 baseline and optimised Ragas evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

BEFORE_PATH = OUTPUT_DIR / "evaluation_before.json"
AFTER_PATH = OUTPUT_DIR / "evaluation_after.json"
SUMMARY_PATH = OUTPUT_DIR / "optimisation_summary.md"

METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def load_results(path: Path) -> dict[str, Any]:
    """Load and validate one evaluation results file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run the baseline and optimised evaluations first."
        )

    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    before = load_results(BEFORE_PATH)
    after = load_results(AFTER_PATH)

    print("W8D2 — Ragas Optimisation Comparison")
    print("=" * 65)

    for label, result in (("BEFORE", before), ("AFTER", after)):
        count = len(result.get("results", []))
        config = result.get("configuration", {})
        print(
            f"{label}: completed={result.get('completed')}, "
            f"samples={count}/{result.get('sample_count')}, "
            f"top-k={config.get('top_k')}"
        )

    if not before.get("completed") or not after.get("completed"):
        raise SystemExit(
            "\nComparison stopped: both evaluations must be complete."
        )

    if len(before.get("results", [])) != 10:
        raise SystemExit("Baseline must contain 10 evaluated samples.")
    if len(after.get("results", [])) != 10:
        raise SystemExit("Optimised run must contain 10 evaluated samples.")

    before_config = before.get("configuration", {})
    after_config = after.get("configuration", {})

    if before_config.get("top_k") == after_config.get("top_k"):
        raise SystemExit(
            "The top-k values are identical. Check the two result files."
        )

    before_scores = before.get("metric_averages", {})
    after_scores = after.get("metric_averages", {})
    before_counts = before.get("successful_metric_scores", {})
    after_counts = after.get("successful_metric_scores", {})

    print()
    print(f"{'Metric':<22}{'Before':>10}{'After':>10}{'Change':>10}")
    print("-" * 52)

    report_lines = [
        "# W8D2 Ragas Optimisation Summary",
        "",
        "## Experiment configuration",
        "",
        f"- Baseline retrieval top-k: {before_config.get('top_k')}",
        f"- Optimised retrieval top-k: {after_config.get('top_k')}",
        f"- Baseline chunk size: {before_config.get('chunk_size')}",
        f"- Optimised chunk size: {after_config.get('chunk_size')}",
        f"- Chunk overlap: {after_config.get('chunk_overlap')}",
        "- Evaluation samples: 10 fixed questions and reference answers",
        "",
        "## Metric comparison",
        "",
        "| Metric | Before | After | Change | Valid scores before/after |",
        "|---|---:|---:|---:|---:|",
    ]

    for metric in METRICS:
        b = before_scores.get(metric)
        a = after_scores.get(metric)
        bc = before_counts.get(metric, 0)
        ac = after_counts.get(metric, 0)

        if b is None or a is None or bc != 10 or ac != 10:
            print(f"{metric:<22}{b!s:>10}{a!s:>10}{'N/A':>10}")
            report_lines.append(
                f"| {metric} | {b} | {a} | N/A | {bc}/10, {ac}/10 |"
            )
            continue

        change = a - b
        print(f"{metric:<22}{b:>10.4f}{a:>10.4f}{change:>+10.4f}")
        report_lines.append(
            f"| {metric} | {b:.4f} | {a:.4f} | {change:+.4f} "
            f"| {bc}/10, {ac}/10 |"
        )

    before_recall = before_scores.get("context_recall")
    after_recall = after_scores.get("context_recall")

    report_lines.extend(["", "## Interpretation", ""])

    if (
        before_recall is not None
        and after_recall is not None
        and before_counts.get("context_recall") == 10
        and after_counts.get("context_recall") == 10
    ):
        recall_change = after_recall - before_recall

        if recall_change > 0:
            conclusion = (
                f"Context recall improved by {recall_change:.4f} "
                f"({before_recall:.4f} to {after_recall:.4f}) after "
                "changing retrieval top-k."
            )
        elif recall_change < 0:
            conclusion = (
                f"Context recall decreased by {abs(recall_change):.4f} "
                f"({before_recall:.4f} to {after_recall:.4f}). "
                "The tested change did not improve context recall."
            )
        else:
            conclusion = (
                f"Context recall did not change ({before_recall:.4f}). "
                "The tested top-k change did not improve this metric."
            )
    else:
        conclusion = (
            "A reliable context-recall comparison is unavailable because "
            "one or both runs lack 10 valid context-recall scores."
        )

    print("\n" + conclusion)
    report_lines.append(conclusion)
   
    report_lines.extend(
    [
        "",
        (
            "Scores are averages from the saved Ragas evaluation outputs. "
            "A positive change means a higher score; a negative change "
            "means a lower score."
        ),
        "",
    ]
)


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    print(f"\nReport saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
