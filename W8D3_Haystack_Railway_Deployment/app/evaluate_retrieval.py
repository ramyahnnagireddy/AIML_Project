
import json
from datetime import UTC, datetime
from pathlib import Path

from haystack_pipeline import build_bm25_pipeline, retrieve_bm25

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"

QUESTIONS = [
    {"id": 1, "question": "What is computer vision?", "expected_source": "computer_vision.pdf"},
    {"id": 2, "question": "What are common applications of computer vision?", "expected_source": "computer_vision.pdf"},
    {"id": 3, "question": "What is deep learning?", "expected_source": "deep_learning.pdf"},
    {"id": 4, "question": "What are neural networks used for?", "expected_source": "deep_learning.pdf"},
    {"id": 5, "question": "What is machine learning?", "expected_source": "machine_learning.pdf"},
    {"id": 6, "question": "What is the difference between supervised and unsupervised learning?", "expected_source": "machine_learning.pdf"},
    {"id": 7, "question": "What is MLOps?", "expected_source": "mlops.pdf"},
    {"id": 8, "question": "Why is model monitoring important in production?", "expected_source": "mlops.pdf"},
    {"id": 9, "question": "What is natural language processing?", "expected_source": "natural_language_processing.pdf"},
    {"id": 10, "question": "What are common natural language processing tasks?", "expected_source": "natural_language_processing.pdf"},
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline, document_store = build_bm25_pipeline()

    print(f"Indexed documents: {document_store.count_documents()}")
    print("Retriever: BM25")
    print(f"Evaluation questions: {len(QUESTIONS)}\n")

    results = []

    for item in QUESTIONS:
        retrieved = retrieve_bm25(pipeline, item["question"], top_k=3)
        sources = [doc["source"] for doc in retrieved]

        relevant_count = sum(
            source == item["expected_source"] for source in sources
        )
        precision_at_3 = relevant_count / 3

        result = {
            **item,
            "retrieved_sources": sources,
            "precision_at_3": precision_at_3,
            "top_1_correct": bool(
                sources and sources[0] == item["expected_source"]
            ),
            "manual_review": "Pending",
        }
        results.append(result)

        print(f"Q{item['id']}: {item['question']}")
        print(f"  Expected: {item['expected_source']}")
        print(f"  Retrieved: {sources}")
        print(f"  Precision@3: {precision_at_3:.3f}")
        print(f"  Top-1 correct: {result['top_1_correct']}\n")

    mean_precision = sum(
        item["precision_at_3"] for item in results
    ) / len(results)
    top_1_accuracy = sum(
        item["top_1_correct"] for item in results
    ) / len(results)

    report = {
        "retriever": "BM25",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "document_count": document_store.count_documents(),
        "question_count": len(results),
        "mean_precision_at_3": mean_precision,
        "top_1_accuracy": top_1_accuracy,
        "results": results,
    }

    output_path = OUTPUT_DIR / "bm25_evaluation.json"
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Mean Precision@3: {mean_precision:.3f}")
    print(f"Top-1 accuracy: {top_1_accuracy:.1%}")
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
