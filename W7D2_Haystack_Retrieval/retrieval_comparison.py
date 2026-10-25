from pathlib import Path
import json


OUTPUTS_DIR = Path("outputs")

EXPECTED_SOURCES = [
    "machine_learning.pdf",
    "machine_learning.pdf",
    "deep_learning.pdf",
    "deep_learning.pdf",
    "natural_language_processing.pdf",
    "natural_language_processing.pdf",
    "computer_vision.pdf",
    "computer_vision.pdf",
    "mlops.pdf",
    "mlops.pdf",
]


def load_results(filename):
    path = OUTPUTS_DIR / filename

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate(results):
    evaluation = []

    for index, item in enumerate(results):
        expected = EXPECTED_SOURCES[index]

        retrieved = item["retrieved_documents"]

        ranks = {
            document["source_file"]: document["rank"]
            for document in retrieved
        }

        top1_source = retrieved[0]["source_file"]

        evaluation.append(
            {
                "question_number": index + 1,
                "question": item["question"],
                "expected_source": expected,
                "top1_source": top1_source,
                "top1_hit": top1_source == expected,
                "expected_rank": ranks.get(expected),
                "top3_hit": expected in ranks,
            }
        )

    return evaluation


def calculate_metrics(evaluation):
    total = len(evaluation)

    top1_hits = sum(
        item["top1_hit"]
        for item in evaluation
    )

    top3_hits = sum(
        item["top3_hit"]
        for item in evaluation
    )

    return {
        "total_questions": total,
        "top1_hits": top1_hits,
        "top1_precision": round(top1_hits / total, 4),
        "top3_hits": top3_hits,
        "top3_hit_rate": round(top3_hits / total, 4),
    }


def print_method_results(name, evaluation, metrics):
    print("\n" + "=" * 80)
    print(f"{name} EVALUATION")
    print("=" * 80)

    for item in evaluation:
        rank = item["expected_rank"]

        print(
            f"Q{item['question_number']}: "
            f"expected={item['expected_source']} | "
            f"rank={rank} | "
            f"top1={'PASS' if item['top1_hit'] else 'MISS'}"
        )

    print("\nMetrics:")
    print(
        f"Top-1 Precision: "
        f"{metrics['top1_hits']}/{metrics['total_questions']} "
        f"({metrics['top1_precision'] * 100:.1f}%)"
    )

    print(
        f"Top-3 Hit Rate: "
        f"{metrics['top3_hits']}/{metrics['total_questions']} "
        f"({metrics['top3_hit_rate'] * 100:.1f}%)"
    )


def main():
    bm25_results = load_results("bm25_results.json")
    dense_results = load_results("dense_results.json")

    if len(bm25_results) != 10:
        raise ValueError("BM25 results must contain 10 questions.")

    if len(dense_results) != 10:
        raise ValueError("Dense results must contain 10 questions.")

    bm25_evaluation = evaluate(bm25_results)
    dense_evaluation = evaluate(dense_results)

    bm25_metrics = calculate_metrics(bm25_evaluation)
    dense_metrics = calculate_metrics(dense_evaluation)

    comparison = {
        "evaluation_method": (
            "Manual relevance mapping against the expected source PDF "
            "for the same 10 questions."
        ),
        "bm25": {
            "metrics": bm25_metrics,
            "questions": bm25_evaluation,
        },
        "dense": {
            "metrics": dense_metrics,
            "questions": dense_evaluation,
        },
    }

    output_path = OUTPUTS_DIR / "retrieval_comparison.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            comparison,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print_method_results(
        "BM25",
        bm25_evaluation,
        bm25_metrics,
    )

    print_method_results(
        "DENSE",
        dense_evaluation,
        dense_metrics,
    )

    print("\n" + "=" * 80)
    print("BM25 VS DENSE COMPARISON")
    print("=" * 80)

    print(
        f"BM25 Top-1 Precision: "
        f"{bm25_metrics['top1_precision'] * 100:.1f}%"
    )

    print(
        f"Dense Top-1 Precision: "
        f"{dense_metrics['top1_precision'] * 100:.1f}%"
    )

    print(
        f"BM25 Top-3 Hit Rate: "
        f"{bm25_metrics['top3_hit_rate'] * 100:.1f}%"
    )

    print(
        f"Dense Top-3 Hit Rate: "
        f"{dense_metrics['top3_hit_rate'] * 100:.1f}%"
    )

    print(f"\nSaved comparison evidence to: {output_path}")


if __name__ == "__main__":
    main()
