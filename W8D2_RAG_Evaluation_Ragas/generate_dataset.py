
"""Create 10 reference Q&A pairs for W8D2 Ragas evaluation."""

from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "qa_dataset.json"

QA_PAIRS: list[dict[str, str]] = [
    {
        "id": "Q01",
        "user_input": "What is Retrieval-Augmented Generation (RAG)?",
        "reference": (
            "RAG combines document retrieval with a language model to answer "
            "questions using external information."
        ),
    },
    {
        "id": "Q02",
        "user_input": "What are the main stages of a RAG pipeline?",
        "reference": (
            "The main stages are document indexing, retrieval of relevant "
            "information, and answer generation."
        ),
    },
    {
        "id": "Q03",
        "user_input": "What is document chunking, and why is it used in RAG?",
        "reference": (
            "Chunking divides long documents into smaller text segments before "
            "they are embedded and indexed."
        ),
    },
    {
        "id": "Q04",
        "user_input": (
            "What is the trade-off between small and large document chunks?"
        ),
        "reference": (
            "Small chunks can improve retrieval precision but may omit useful "
            "surrounding information. Larger chunks preserve more context but "
            "may include unrelated information."
        ),
    },
    {
        "id": "Q05",
        "user_input": (
            "Why is chunk overlap used, and what is a disadvantage of excessive overlap?"
        ),
        "reference": (
            "Chunk overlap repeats some text between adjacent chunks and helps "
            "preserve facts across chunk boundaries. Excessive overlap can create "
            "redundancy and increase storage and retrieval costs."
        ),
    },
    {
        "id": "Q06",
        "user_input": (
            "What are embeddings, and how do they support semantic search?"
        ),
        "reference": (
            "Embeddings represent text as numerical vectors. Semantic search uses "
            "vector similarity to find information related to a question, even "
            "when the wording differs."
        ),
    },
    {
        "id": "Q07",
        "user_input": "What is the role of ChromaDB in a RAG system?",
        "reference": (
            "ChromaDB stores text embeddings and can associate them with documents "
            "and metadata. It supports similarity-based retrieval from an indexed "
            "collection."
        ),
    },
    {
        "id": "Q08",
        "user_input": (
            "How does the top-k setting affect the documents retrieved by a RAG pipeline?"
        ),
        "reference": (
            "Top-k controls how many chunks are returned for a question. A small "
            "value may omit relevant evidence, while a larger value may improve "
            "coverage but introduce irrelevant chunks."
        ),
    },
    {
        "id": "Q09",
        "user_input": (
            "What is faithfulness in RAG evaluation, and how does it differ from answer relevancy?"
        ),
        "reference": (
            "Faithfulness measures whether answer claims are supported by retrieved "
            "context. Answer relevancy measures whether the response appropriately "
            "addresses the user's question."
        ),
    },
    {
        "id": "Q10",
        "user_input": (
            "How do context precision and context recall differ in RAG evaluation?"
        ),
        "reference": (
            "Context precision measures whether relevant retrieved chunks are ranked "
            "ahead of irrelevant chunks. Context recall measures how much of the "
            "information needed to support the reference answer is present in the "
            "retrieved context."
        ),
    },
]


def main() -> None:
    """Validate and save the reference dataset."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(QA_PAIRS) != 10:
        raise ValueError(f"Expected 10 Q&A pairs, found {len(QA_PAIRS)}.")

    for item in QA_PAIRS:
        if not item["user_input"].strip() or not item["reference"].strip():
            raise ValueError(f"Empty question or reference in {item['id']}.")

    payload = {
        "description": (
            "Ten reference Q&A pairs derived from the W8D2 RAG knowledge base. "
            "The RAG-generated responses and retrieved contexts are recorded "
            "during evaluation."
        ),
        "sample_count": len(QA_PAIRS),
        "samples": QA_PAIRS,
    }

    OUTPUT_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("W8D2 reference dataset created successfully.")
    print(f"Samples: {len(QA_PAIRS)}")
    print(f"Output: {OUTPUT_FILE}")
    for item in QA_PAIRS:
        print(f"{item['id']}: {item['user_input']}")


if __name__ == "__main__":
    main()
    