from pathlib import Path

import chromadb
from langchain_ollama import OllamaEmbeddings


# Paths
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

# Configuration
COLLECTION_NAME = "w6d4_rag_documents"
EMBEDDING_MODEL = "nomic-embed-text:latest"


def main():
    print("=" * 60)
    print("W6D4 - ChromaDB Similarity Search")
    print("=" * 60)

    # Connect to the persistent ChromaDB database
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Load the existing collection
    collection = client.get_collection(name=COLLECTION_NAME)

    # Create the same embedding model used during indexing
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    query = input("\nEnter your search query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    # Convert query into an embedding
    query_embedding = embeddings.embed_query(query)

    # Perform similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "distances"],
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    print("\nTop 3 Similar Documents")
    print("-" * 60)

    for index, (document, distance) in enumerate(
        zip(documents, distances), start=1
    ):
        print(f"\nResult {index}")
        print(f"Distance: {distance:.4f}")
        print(f"Document: {document}")

    print("\nSimilarity search completed successfully.")


if __name__ == "__main__":
    main()
    