import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"


def main():
    bm25_path = OUTPUT_DIR / "bm25_evaluation.json"
    dense_path = OUTPUT_DIR / "dense_evaluation.json"

    with bm25_path.open(encoding="utf-8") as file:
        bm25 = json.load(file)

    with dense_path.open(encoding="utf-8") as file:
        dense = json.load(file)

    comparison = {
        "questions_evaluated": 10,
        "documents_indexed": 5,
        "metrics": {
            "top_1_accuracy": {
                "bm25": bm25["top_1_accuracy"],
                "dense": dense["top_1_accuracy"],
            },
            "mean_precision_at_3": {
                "bm25": bm25["mean_precision_at_3"],
                "dense": dense["mean_precision_at_3"],
            },
        },
        "conclusion": (
            "Dense retrieval performed better on top-1 accuracy for this "
            "10-question evaluation. Both methods had the same mean "
            "Precision@3."
        ),
    }

    output_path = OUTPUT_DIR / "retrieval_comparison.json"
    output_path.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    print("BM25 vs Dense Retrieval")
    print("-" * 32)
    print(
        f"BM25 top-1 accuracy: "
        f"{bm25['top_1_accuracy']:.1%}"
    )
    print(
        f"Dense top-1 accuracy: "
        f"{dense['top_1_accuracy']:.1%}"
    )
    print(
        f"BM25 Precision@3: "
        f"{bm25['mean_precision_at_3']:.3f}"
    )
    print(
        f"Dense Precision@3: "
        f"{dense['mean_precision_at_3']:.3f}"
    )
    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
