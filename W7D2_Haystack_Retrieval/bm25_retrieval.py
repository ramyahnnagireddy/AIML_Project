from pathlib import Path
import json

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


DOCUMENTS_DIR = Path("documents")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

QUESTIONS = [
    "What is supervised learning?",
    "Which metrics are used for classification evaluation?",
    "What are convolutional neural networks commonly used for?",
    "How are weights updated during neural network training?",
    "What are text embeddings?",
    "What applications use natural language processing?",
    "What does object detection identify?",
    "What is image segmentation?",
    "What is MLOps used for?",
    "What information can MLflow track?",
]


def load_documents():
    document_store = InMemoryDocumentStore()

    converter = PyPDFToDocument()

    pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if len(pdf_paths) != 5:
        raise ValueError(
            f"Expected 5 PDF documents, but found {len(pdf_paths)}."
        )

    all_documents = []

    for pdf_path in pdf_paths:
        result = converter.run(sources=[pdf_path])
        documents = result["documents"]

        for document in documents:
            document.meta["source_file"] = pdf_path.name

        all_documents.extend(documents)

    document_store.write_documents(all_documents)

    return document_store


def build_pipeline(document_store):
    retriever = InMemoryBM25Retriever(
        document_store=document_store,
        top_k=3,
    )

    pipeline = Pipeline()
    pipeline.add_component("retriever", retriever)

    return pipeline


def run_retrieval(pipeline):
    results = []

    for question in QUESTIONS:
        output = pipeline.run(
            {
                "retriever": {
                    "query": question,
                }
            }
        )

        documents = output["retriever"]["documents"]

        retrieved = []

        for rank, document in enumerate(documents, start=1):
            retrieved.append(
                {
                    "rank": rank,
                    "source_file": document.meta.get(
                        "source_file",
                        "unknown",
                    ),
                    "content": document.content,
                    "score": document.score,
                }
            )

        results.append(
            {
                "question": question,
                "retrieved_documents": retrieved,
            }
        )

    return results


def save_results(results):
    output_path = OUTPUTS_DIR / "bm25_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved evidence to: {output_path}")


def print_results(results):
    print("\n" + "=" * 80)
    print("BM25 RETRIEVAL RESULTS")
    print("=" * 80)

    for item in results:
        print(f"\nQuestion: {item['question']}")

        for document in item["retrieved_documents"]:
            print(
                f"  Rank {document['rank']} | "
                f"{document['source_file']} | "
                f"score={document['score']:.4f}"
            )


def main():
    print("Loading 5 PDF documents...")

    document_store = load_documents()

    print(
        f"Indexed documents: {document_store.count_documents()}"
    )

    pipeline = build_pipeline(document_store)

    results = run_retrieval(pipeline)

    print_results(results)
    save_results(results)

    print("\nBM25 retrieval completed successfully.")


if __name__ == "__main__":
    main()
    