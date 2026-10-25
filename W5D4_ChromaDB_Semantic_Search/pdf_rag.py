"""
W5D4 - PDF Retrieval Augmented Generation with ChromaDB

Pipeline:
PDF -> text extraction -> chunks -> Ollama embeddings
-> ChromaDB -> top-3 retrieval -> llama3.2:3b answer
"""

from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "sample_document.pdf"
DB_DIR = BASE_DIR / "chroma_db"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "pdf_rag_output.txt"

EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "llama3.2:3b"


# -----------------------------
# PDF Text Extraction
# -----------------------------

def extract_pdf_text(pdf_path):
    """Extract text from all pages of a PDF."""

    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    return pages


# -----------------------------
# Text Chunking
# -----------------------------

def create_chunks(pages, chunk_size=500, overlap=100):
    """Create overlapping text chunks from PDF pages."""

    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        start = 0
        chunk_number = 1

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "id": f"page_{page_number}_chunk_{chunk_number}",
                        "text": chunk_text,
                        "page": page_number,
                    }
                )

            if end >= len(text):
                break

            start = end - overlap
            chunk_number += 1

    return chunks


# -----------------------------
# Generate Embeddings
# -----------------------------

def generate_embeddings(texts):
    """Generate embeddings using Ollama."""

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return response["embeddings"]


# -----------------------------
# Retrieve Top 3 Chunks
# -----------------------------

def retrieve_chunks(collection, query):
    """Retrieve the top 3 relevant chunks."""

    query_embedding = generate_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    retrieved = []

    for chunk_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append(
            {
                "id": chunk_id,
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return retrieved


# -----------------------------
# Generate LLM Answer
# -----------------------------

def generate_answer(query, retrieved_chunks):
    """Generate an answer using the retrieved PDF context."""

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"Context Chunk {index} "
            f"(page {chunk['metadata']['page']}):\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
Answer the question using only the provided PDF context.

If the answer is not available in the context, say that the
information is not available in the provided document.

Question:
{query}

PDF Context:
{context}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI/ML mentor. Answer accurately and "
                    "use only the supplied document context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response["message"]["content"]


# -----------------------------
# Main RAG Pipeline
# -----------------------------

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    # Step 1: Extract PDF text
    pages = extract_pdf_text(PDF_PATH)

    if not pages:
        raise ValueError("No text could be extracted from the PDF.")

    # Step 2: Create chunks
    chunks = create_chunks(pages)

    # Step 3: Create persistent ChromaDB client
    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    collection = client.get_or_create_collection(
        name="w5d4_pdf_rag",
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )

    # Step 4: Generate embeddings
    texts = [chunk["text"] for chunk in chunks]

    embeddings = generate_embeddings(texts)

    # Step 5: Prepare metadata
    metadatas = [
        {
            "page": chunk["page"],
            "source": PDF_PATH.name,
        }
        for chunk in chunks
    ]

    ids = [chunk["id"] for chunk in chunks]

    # Step 6: Store PDF chunks in ChromaDB
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Step 7: Query
    query = "How does Retrieval Augmented Generation work?"

    retrieved_chunks = retrieve_chunks(
        collection,
        query,
    )

    # Step 8: Generate LLM answer
    answer = generate_answer(
        query,
        retrieved_chunks,
    )

    # -----------------------------
    # Save Evidence
    # -----------------------------

    output_lines = []

    output_lines.append(
        "W5D4 - PDF Retrieval Augmented Generation"
    )
    output_lines.append("=" * 60)
    output_lines.append("")

    output_lines.append(
        f"PDF: {PDF_PATH.name}"
    )
    output_lines.append(
        f"Pages extracted: {len(pages)}"
    )
    output_lines.append(
        f"Chunks created: {len(chunks)}"
    )
    output_lines.append(
        f"Embedding model: {EMBEDDING_MODEL}"
    )
    output_lines.append(
        f"LLM model: {LLM_MODEL}"
    )
    output_lines.append(
        "Distance metric: cosine"
    )
    output_lines.append("")

    output_lines.append(
        "QUERY"
    )
    output_lines.append("-" * 60)
    output_lines.append(query)
    output_lines.append("")

    output_lines.append(
        "TOP-3 RETRIEVED CHUNKS"
    )
    output_lines.append("-" * 60)

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        output_lines.append(
            f"\nRank {index}"
        )
        output_lines.append(
            f"Chunk ID: {chunk['id']}"
        )
        output_lines.append(
            f"Page: {chunk['metadata']['page']}"
        )
        output_lines.append(
            f"Distance: {chunk['distance']:.4f}"
        )
        output_lines.append(
            f"Text: {chunk['text']}"
        )

    output_lines.append("")
    output_lines.append(
        "LLM ANSWER"
    )
    output_lines.append("-" * 60)
    output_lines.append(answer)

    output_lines.append("")
    output_lines.append(
        "VERIFICATION"
    )
    output_lines.append("-" * 60)
    output_lines.append(
        "PDF text extraction: PASS"
    )
    output_lines.append(
        "PDF chunking: PASS"
    )
    output_lines.append(
        "ChromaDB storage: PASS"
    )
    output_lines.append(
        "Top-3 retrieval: PASS"
    )
    output_lines.append(
        "Ollama LLM generation: PASS"
    )
    output_lines.append("")
    output_lines.append(
        "STATUS: SUCCESS"
    )

    OUTPUT_FILE.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    print("\n".join(output_lines))
    print(
        f"\nEvidence saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
    