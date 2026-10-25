

import json
from pathlib import Path

from dense_retrieval import build_dense_pipeline, retrieve_dense

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"

QUESTIONS = [
    ("What is computer vision?", "computer_vision.pdf"),
    ("What are common applications of computer vision?", "computer_vision.pdf"),
    ("What is deep learning?", "deep_learning.pdf"),
    ("What are neural networks used for?", "deep_learning.pdf"),
    ("What is machine learning?", "machine_learning.pdf"),
    (
        "What is the difference between supervised and unsupervised learning?",
        "machine_learning.pdf",
    ),
    ("What is MLOps?", "mlops.pdf"),
    ("Why is model monitoring important in production?", "mlops.pdf"),
    ("What is natural language processing?", "natural_language_processing.pdf"),
    (
        "What are common natural language processing tasks?",
        "natural_language_processing.pdf",
    ),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline, store = build_dense_pipeline()
    results = []
    top_1_correct = 0
    precision_values = []

    print(f"Indexed documents: {store.count_documents()}")
    print("Retriever: Dense")
    print(f"Evaluation questions: {len(QUESTIONS)}\n")

    for index, (question, expected_source) in enumerate(QUESTIONS, start=1):
        retrieved = retrieve_dense(pipeline, question, top_k=3)
        sources = [item["source"] for item in retrieved]

        hits = sum(source == expected_source for source in sources)
        precision_at_3 = hits / 3
        top_1 = bool(sources and sources[0] == expected_source)

        top_1_correct += int(top_1)
        precision_values.append(precision_at_3)

        result = {
            "question": question,
            "expected_source": expected_source,
            "retrieved_sources": sources,
            "precision_at_3": round(precision_at_3, 4),
            "top_1_correct": top_1,
            "retrieved_documents": retrieved,
        }
        results.append(result)

        print(f"Q{index}: {question}")
        print(f"  Expected: {expected_source}")
        print(f"  Retrieved: {sources}")
        print(f"  Precision@3: {precision_at_3:.3f}")
        print(f"  Top-1 correct: {top_1}\n")

    report = {
        "retriever": "Dense",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "indexed_documents": store.count_documents(),
        "question_count": len(QUESTIONS),
        "mean_precision_at_3": round(
            sum(precision_values) / len(precision_values), 4
        ),
        "top_1_accuracy": round(top_1_correct / len(QUESTIONS), 4),
        "results": results,
    }

    report_path = OUTPUT_DIR / "dense_evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Mean Precision@3: {report['mean_precision_at_3']:.3f}")
    print(f"Top-1 accuracy: {report['top_1_accuracy']:.1%}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
