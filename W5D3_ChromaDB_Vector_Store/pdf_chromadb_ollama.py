import chromadb
import ollama
from pypdf import PdfReader


PDF_PATH = "documents/ai_ml_reference.pdf"
COLLECTION_NAME = "pdf_ai_ml_documents"
MODEL_NAME = "llama3.2:3b"


def extract_pdf_text(pdf_path):
    """Extract text from all pages of the PDF."""
    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text.strip())

    return "\n".join(pages)


def create_chunks(text, chunk_size=80):
    """Split extracted text into simple word-based chunks."""
    words = text.split()
    chunks = []

    for start in range(0, len(words), chunk_size):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


def main():
    print("=" * 70)
    print("PDF -> CHROMADB -> OLLAMA RAG PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Extract PDF text
    # ---------------------------------------------------------
    text = extract_pdf_text(PDF_PATH)

    print("\n1. PDF TEXT EXTRACTION")
    print("-" * 70)
    print(f"PDF: {PDF_PATH}")
    print(f"Extracted characters: {len(text)}")

    # ---------------------------------------------------------
    # 2. Create chunks
    # ---------------------------------------------------------
    chunks = create_chunks(text)

    print("\n2. DOCUMENT CHUNKING")
    print("-" * 70)
    print(f"Number of chunks: {len(chunks)}")

    # ---------------------------------------------------------
    # 3. Store chunks in ChromaDB
    # ---------------------------------------------------------
    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )

    ids = [f"pdf_chunk_{i + 1}" for i in range(len(chunks))]

    metadatas = [
        {
            "source": "ai_ml_reference.pdf",
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )

    print("\n3. CHROMADB STORAGE")
    print("-" * 70)
    print(f"Collection: {collection.name}")
    print(f"Stored chunks: {collection.count()}")

    # ---------------------------------------------------------
    # 4. Similarity search
    # ---------------------------------------------------------
    query = "How can machine learning models be improved and evaluated?"

    results = collection.query(
        query_texts=[query],
        n_results=min(3, len(chunks)),
    )

    retrieved_documents = results["documents"][0]
    distances = results["distances"][0]
    retrieved_metadata = results["metadatas"][0]

    print("\n4. TOP-3 SIMILARITY RETRIEVAL")
    print("-" * 70)
    print(f"Query: {query}")

    for i, (document, distance, metadata) in enumerate(
        zip(retrieved_documents, distances, retrieved_metadata),
        start=1,
    ):
        print(f"\nRetrieved Chunk {i}")
        print(f"Cosine distance: {distance:.4f}")
        print(f"Metadata: {metadata}")
        print(f"Text: {document}")

    # ---------------------------------------------------------
    # 5. Build context for Ollama
    # ---------------------------------------------------------
    context = "\n\n".join(
        [
            f"Context {i + 1}:\n{document}"
            for i, document in enumerate(retrieved_documents)
        ]
    )

    prompt = f"""
Answer the question using only the provided context.

Question:
{query}

Context:
{context}

Give a concise and beginner-friendly answer.
"""

    # ---------------------------------------------------------
    # 6. Send retrieved context to Ollama
    # ---------------------------------------------------------
    print("\n5. OLLAMA GENERATION")
    print("-" * 70)
    print(f"Model: {MODEL_NAME}")

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI/ML mentor. Answer questions using "
                    "only the supplied context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer = response["message"]["content"]

    print("\nGenerated Answer:")
    print(answer)

    # ---------------------------------------------------------
    # 7. Save evidence
    # ---------------------------------------------------------
    output_path = "outputs/pdf_ollama_rag_output.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("PDF -> CHROMADB -> OLLAMA RAG PIPELINE\n")
        file.write("=" * 70 + "\n\n")

        file.write(f"PDF: {PDF_PATH}\n")
        file.write(f"Collection: {collection.name}\n")
        file.write(f"Stored chunks: {collection.count()}\n")
        file.write(f"Query: {query}\n\n")

        file.write("TOP-3 RETRIEVED CHUNKS\n")
        file.write("-" * 70 + "\n")

        for i, (document, distance, metadata) in enumerate(
            zip(retrieved_documents, distances, retrieved_metadata),
            start=1,
        ):
            file.write(f"\nChunk {i}\n")
            file.write(f"Cosine distance: {distance:.4f}\n")
            file.write(f"Metadata: {metadata}\n")
            file.write(f"Text: {document}\n")

        file.write("\n\nOLLAMA ANSWER\n")
        file.write("-" * 70 + "\n")
        file.write(answer)
        file.write("\n")

    print(f"\nEvidence saved to: {output_path}")

    print("\n" + "=" * 70)
    print("PDF + CHROMADB + OLLAMA PIPELINE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
    