from pathlib import Path
import json

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
OUTPUTS_DIR = BASE_DIR / "outputs"

QUESTIONS = [
    "What is machine learning?",
    "What is supervised learning?",
    "What is deep learning?",
    "What is natural language processing?",
    "What is a neural network?",
    "What is retrieval augmented generation?",
    "What is model training?",
    "What is classification?",
    "What is an embedding?",
    "What is the purpose of evaluation in machine learning?",
]


def get_pdf_files():
    """Return the five PDF files used for the experiment."""
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if len(pdf_files) != 5:
        raise ValueError(
            f"Expected exactly 5 PDF files, but found {len(pdf_files)}."
        )

    return pdf_files


def build_bm25_store(pdf_files):
    """Convert PDFs and store their documents for BM25 retrieval."""
    document_store = InMemoryDocumentStore()

    converter = PyPDFToDocument()
    writer = DocumentWriter(document_store=document_store)

    pipeline = Pipeline()

    pipeline.add_component("converter", converter)
    pipeline.add_component("writer", writer)

    pipeline.connect(
        "converter.documents",
        "writer.documents",
    )

    pipeline.run(
        {
            "converter": {
                "sources": pdf_files,
            }
        }
    )

    return document_store


def run_bm25(document_store):
    """Run all ten questions using BM25 retrieval."""
    retriever = InMemoryBM25Retriever(
        document_store=document_store,
        top_k=3,
    )

    results = []

    print("\n" + "=" * 70)
    print("BM25 RETRIEVAL")
    print("=" * 70)

    for question_number, question in enumerate(QUESTIONS, start=1):
        output = retriever.run(query=question)
        documents = output["documents"]

        print(f"\nQuestion {question_number}: {question}")

        question_results = []

        for rank, document in enumerate(documents, start=1):
            source = Path(
                document.meta.get("file_path", "unknown")
            ).name

            score = document.score

            print(
                f"  {rank}. {source} | "
                f"score={score:.4f}"
            )

            question_results.append(
                {
                    "rank": rank,
                    "source": source,
                    "score": score,
                    "preview": document.content[:200],
                }
            )

        results.append(
            {
                "question_number": question_number,
                "question": question,
                "results": question_results,
            }
        )

    return results


def build_dense_store(pdf_files):
    """Convert PDFs, generate embeddings, and store them."""
    document_store = InMemoryDocumentStore(
        embedding_similarity_function="cosine"
    )

    converter = PyPDFToDocument()

    document_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    writer = DocumentWriter(
        document_store=document_store
    )

    pipeline = Pipeline()

    pipeline.add_component(
        "converter",
        converter,
    )

    pipeline.add_component(
        "document_embedder",
        document_embedder,
    )

    pipeline.add_component(
        "writer",
        writer,
    )

    pipeline.connect(
        "converter.documents",
        "document_embedder.documents",
    )

    pipeline.connect(
        "document_embedder.documents",
        "writer.documents",
    )

    pipeline.run(
        {
            "converter": {
                "sources": pdf_files,
            }
        }
    )

    return document_store


def run_dense(document_store):
    """Run all ten questions using dense embedding retrieval."""
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=3,
    )

    results = []

    print("\n" + "=" * 70)
    print("DENSE RETRIEVAL")
    print("=" * 70)

    for question_number, question in enumerate(QUESTIONS, start=1):
        embedding_output = text_embedder.run(
            text=question
        )

        query_embedding = embedding_output["embedding"]

        output = retriever.run(
            query_embedding=query_embedding
        )

        documents = output["documents"]

        print(f"\nQuestion {question_number}: {question}")

        question_results = []

        for rank, document in enumerate(documents, start=1):
            source = Path(
                document.meta.get("file_path", "unknown")
            ).name

            score = document.score

            print(
                f"  {rank}. {source} | "
                f"score={score:.4f}"
            )

            question_results.append(
                {
                    "rank": rank,
                    "source": source,
                    "score": score,
                    "preview": document.content[:200],
                }
            )

        results.append(
            {
                "question_number": question_number,
                "question": question,
                "results": question_results,
            }
        )

    return results


def save_results(pdf_files, bm25_results, dense_results):
    """Save retrieval results as W7D1 evidence."""
    OUTPUTS_DIR.mkdir(exist_ok=True)

    output_file = OUTPUTS_DIR / "retrieval_results.json"

    data = {
        "experiment": "W7D1 Haystack Pipeline Architecture",
        "pdf_count": len(pdf_files),
        "question_count": len(QUESTIONS),
        "pdf_files": [
            pdf.name for pdf in pdf_files
        ],
        "questions": QUESTIONS,
        "bm25": bm25_results,
        "dense": dense_results,
    }

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            default=str,
        )

    return output_file


def main():
    print("=" * 70)
    print("W7D1: HAYSTACK PIPELINE ARCHITECTURE")
    print("=" * 70)

    pdf_files = get_pdf_files()

    print("\nPDF documents:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    print(f"\nTotal PDFs: {len(pdf_files)}")
    print(f"Total questions: {len(QUESTIONS)}")

    print("\nBuilding BM25 document store...")
    bm25_store = build_bm25_store(pdf_files)

    bm25_results = run_bm25(bm25_store)

    print("\nBuilding dense document store...")
    dense_store = build_dense_store(pdf_files)

    dense_results = run_dense(dense_store)

    output_file = save_results(
        pdf_files,
        bm25_results,
        dense_results,
    )

    print("\n" + "=" * 70)
    print("W7D1 RETRIEVAL EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Evidence file: {output_file}")


if __name__ == "__main__":
    main()
    