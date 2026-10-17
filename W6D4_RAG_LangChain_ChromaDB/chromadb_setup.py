from pathlib import Path

import chromadb
from langchain_ollama import OllamaEmbeddings


# Paths
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"


# Configuration
COLLECTION_NAME = "w6d4_rag_documents"
EMBEDDING_MODEL = "nomic-embed-text:latest"


def get_embeddings():
    """Create the Ollama embedding model."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def create_chroma_collection():
    """Create or load the ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "W6D4 LangChain RAG document collection"},
    )

    return client, collection


def add_documents(collection, embeddings):
    """Add sample documents with their embeddings."""

    documents = [
        (
            "doc1",
            "Machine learning enables computers to learn patterns "
            "from data and make predictions without being explicitly programmed."
        ),
        (
            "doc2",
            "Deep learning uses neural networks with multiple layers "
            "to learn complex patterns from large datasets."
        ),
        (
            "doc3",
            "Natural language processing allows computers to understand "
            "and generate human language."
        ),
        (
            "doc4",
            "Retrieval Augmented Generation combines document retrieval "
            "with language generation to provide answers grounded in external information."
        ),
        (
            "doc5",
            "Vector databases store numerical representations called embeddings "
            "and support similarity search between related pieces of information."
        ),
    ]

    ids = [item[0] for item in documents]
    texts = [item[1] for item in documents]

    vectors = embeddings.embed_documents(texts)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=vectors,
    )

    return len(documents)


def main():
    print("=" * 60)
    print("W6D4 - ChromaDB Vector Store Setup")
    print("=" * 60)

    print(f"\nEmbedding model: {EMBEDDING_MODEL}")
    print(f"ChromaDB path: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")

    embeddings = get_embeddings()
    _, collection = create_chroma_collection()

    count = add_documents(collection, embeddings)

    print(f"\nAdded/updated documents: {count}")
    print(f"Collection document count: {collection.count()}")

    print("\nChromaDB setup completed successfully.")


if __name__ == "__main__":
    main()
    