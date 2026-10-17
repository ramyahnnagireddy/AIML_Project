from pathlib import Path
import json

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)


DOCUMENTS_DIR = Path("documents")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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


def create_document_embeddings(document_store):
    embedder = SentenceTransformersDocumentEmbedder(
        model=EMBEDDING_MODEL
    )

    embedder.warm_up()

    documents = document_store.filter_documents()

    result = embedder.run(documents)

    document_store.delete_all_documents()
    document_store.write_documents(result["documents"])

    return document_store


def build_pipeline(document_store):
    text_embedder = SentenceTransformersTextEmbedder(
        model=EMBEDDING_MODEL
    )

    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=3,
    )

    pipeline = Pipeline()

    pipeline.add_component(
        "text_embedder",
        text_embedder,
    )

    pipeline.add_component(
        "retriever",
        retriever,
    )

    pipeline.connect(
        "text_embedder.embedding",
        "retriever.query_embedding",
    )

    return pipeline


def run_retrieval(pipeline):
    results = []

    for question in QUESTIONS:
        output = pipeline.run(
            {
                "text_embedder": {
                    "text": question,
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
    output_path = OUTPUTS_DIR / "dense_results.json"

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
    print("DENSE RETRIEVAL RESULTS")
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

    print("\nCreating document embeddings...")

    document_store = create_document_embeddings(
        document_store
    )

    print(
        f"Embedded documents: {document_store.count_documents()}"
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    pipeline = build_pipeline(document_store)

    results = run_retrieval(pipeline)

    print_results(results)
    save_results(results)

    print("\nDense retrieval completed successfully.")


if __name__ == "__main__":
    main()
    