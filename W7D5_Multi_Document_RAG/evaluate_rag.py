"""Evaluate multi-document RAG outputs and save evidence."""

import json
import re
import sys
from datetime import datetime, timezone

from config import OUTPUTS_DIR


def normalise(text):
    """Convert text to lowercase tokens."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def calculate_scores(question, answer, contexts):
    """Calculate simple context support and answer relevance indicators."""
    answer_tokens = set(normalise(answer))
    context_tokens = set(normalise(" ".join(contexts)))
    question_tokens = set(normalise(question))

    content_tokens = {
        token for token in answer_tokens if len(token) > 2
    }
    question_content = {
        token for token in question_tokens if len(token) > 2
    }

    support_score = (
        len(content_tokens & context_tokens) / len(content_tokens)
        if content_tokens
        else 0.0
    )

    relevance_score = (
        len(question_content & answer_tokens) / len(question_content)
        if question_content
        else 0.0
    )

    return round(support_score, 4), round(relevance_score, 4)


def main():
    input_file = OUTPUTS_DIR / "rag_results.json"
    output_file = OUTPUTS_DIR / "ragas_evaluation.json"
    report_file = OUTPUTS_DIR / "RAGAS_EVALUATION.md"

    if not input_file.exists():
        print(f"Missing RAG results: {input_file}")
        return 1

    try:
        records = json.loads(input_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read RAG results: {exc}")
        return 1

    if not isinstance(records, list):
        print("Invalid results format: expected a JSON list.")
        return 1

    results = []

    for item in records:
        if not isinstance(item, dict):
            continue

        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        chunks = item.get("retrieved_chunks", [])

        if not isinstance(chunks, list):
            chunks = []

        contexts = [
            str(chunk.get("text", "")).strip()
            for chunk in chunks
            if isinstance(chunk, dict)
            and str(chunk.get("text", "")).strip()
        ]

        if not question or not answer or not contexts:
            continue

        support, relevance = calculate_scores(
            question, answer, contexts
        )

        results.append(
            {
                "question": question,
                "answer": answer,
                "context_count": len(contexts),
                "context_support_indicator": support,
                "question_term_overlap_indicator": relevance,
                "latency_seconds": item.get("latency_seconds"),
            }
        )

    if not results:
        print("No valid evaluation samples were found.")
        return 1

    support_values = [
        item["context_support_indicator"] for item in results
    ]
    relevance_values = [
        item["question_term_overlap_indicator"] for item in results
    ]

    average_support = round(sum(support_values) / len(support_values), 4)
    average_relevance = round(
        sum(relevance_values) / len(relevance_values), 4
    )

    payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(results),
        "evaluation_method": "deterministic lexical indicators",
        "metrics": {
            "average_context_support_indicator": average_support,
            "average_question_term_overlap_indicator": average_relevance,
        },
        "scores": results,
        "limitations": [
            "These lexical indicators are not Ragas Faithfulness or AnswerRelevancy scores.",
            "Token overlap does not establish semantic correctness or factual entailment.",
        ],
    }

    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        output_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        lines = [
            "# W7D5 RAG Evaluation",
            "",
            f"- Evaluation time: {payload['evaluated_at']}",
            f"- Samples evaluated: {len(results)}",
            "- Method: deterministic lexical indicators",
            "",
            "## Aggregate Results",
            "",
            "| Indicator | Average |",
            "|---|---:|",
            f"| Context support indicator | {average_support} |",
            f"| Question-term overlap indicator | {average_relevance} |",
            "",
            "## Per-sample Results",
            "",
            "| Sample | Context support | Question-term overlap |",
            "|---:|---:|---:|",
        ]

        for index, item in enumerate(results, start=1):
            lines.append(
                f"| {index} | {item['context_support_indicator']} "
                f"| {item['question_term_overlap_indicator']} |"
            )

        lines.extend(
            [
                "",
                "## Limitations",
                "",
                "- These are lexical indicators, not Ragas Faithfulness or AnswerRelevancy scores.",
                "- Token overlap does not establish semantic correctness or factual entailment.",
                "- Results should be interpreted as preliminary diagnostics only.",
                "",
            ]
        )

        report_file.write_text("\n".join(lines), encoding="utf-8")

    except OSError as exc:
        print(f"Could not save evaluation evidence: {exc}")
        return 1

    print("\nEvaluation completed.")
    print(f"Samples evaluated: {len(results)}")
    print(f"Average context support indicator: {average_support}")
    print(f"Average question-term overlap indicator: {average_relevance}")
    print(f"JSON evidence: {output_file}")
    print(f"Markdown report: {report_file}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nEvaluation cancelled.")
        sys.exit(130)
        