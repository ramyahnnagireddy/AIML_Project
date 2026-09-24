from pathlib import Path

import chromadb
from langchain_ollama import ChatOllama, OllamaEmbeddings


# Paths
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

# Configuration
COLLECTION_NAME = "w6d4_rag_documents"
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "llama3.2:3b"


def retrieve_documents(query, collection, embeddings, top_k=3):
    """Retrieve the most relevant documents from ChromaDB."""

    query_embedding = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"],
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    return documents, distances


def generate_answer(query, documents, llm):
    """Generate an answer using retrieved documents as context."""

    context = "\n\n".join(
        f"Source {index}: {document}"
        for index, document in enumerate(documents, start=1)
    )

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using only the provided context.
If the context does not contain enough information, clearly say that
the information is not available in the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content


def main():
    print("=" * 60)
    print("W6D4 - LangChain + ChromaDB RAG Pipeline")
    print("=" * 60)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # Embedding model
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    # Local LLM
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )

    print(f"\nEmbedding model: {EMBEDDING_MODEL}")
    print(f"LLM model: {LLM_MODEL}")
    print(f"Documents in collection: {collection.count()}")

    query = input("\nEnter your question: ").strip()

    if not query:
        print("Question cannot be empty.")
        return

    # Step 1: Retrieve relevant documents
    documents, distances = retrieve_documents(
        query,
        collection,
        embeddings,
        top_k=3,
    )

    print("\nRetrieved Context")
    print("-" * 60)

    for index, (document, distance) in enumerate(
        zip(documents, distances),
        start=1,
    ):
        print(f"\nSource {index}")
        print(f"Distance: {distance:.4f}")
        print(f"Document: {document}")

    # Step 2: Generate answer from retrieved context
    answer = generate_answer(
        query,
        documents,
        llm,
    )

    print("\nGenerated Answer")
    print("-" * 60)
    print(answer)

    print("\nRAG pipeline completed successfully.")


if __name__ == "__main__":
    main()
    