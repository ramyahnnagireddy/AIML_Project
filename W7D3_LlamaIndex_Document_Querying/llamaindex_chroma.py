
import json
import time
from pathlib import Path

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "chroma_query_results.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

QUERIES = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is the difference between classification and regression?",
    "What is unsupervised learning?",
    "What are convolutional neural networks used for?",
    "What is overfitting?",
    "What is tokenization in NLP?",
    "What are text embeddings?",
    "What is retrieval-augmented generation?",
    "How does ChromaDB help a RAG system?",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = SimpleDirectoryReader(
        input_dir=str(DOCUMENTS_DIR),
        required_exts=[".txt"],
    ).load_data()

    if not documents:
        raise ValueError(f"No .txt documents found in {DOCUMENTS_DIR}")

    print(f"Loaded {len(documents)} documents.")

    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
    )
    Settings.llm = Ollama(
        model="qwen2.5:3b",
        base_url="http://localhost:11434",
        request_timeout=240.0,
        additional_kwargs={"num_ctx": 1024},
    )
    Settings.chunk_size = 256
    Settings.chunk_overlap = 40

    print("Opening persistent ChromaDB database...")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Use a fresh collection for a repeatable experiment.
    try:
        chroma_client.delete_collection("w7d3_llamaindex_documents")
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        "w7d3_llamaindex_documents"
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    print("Building ChromaDB-backed vector index...")
    index_start = time.perf_counter()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    indexing_seconds = time.perf_counter() - index_start

    query_engine = index.as_query_engine(similarity_top_k=3)
    results = []

    for number, question in enumerate(QUERIES, start=1):
        print(f"\nQuery {number}/10: {question}")
        query_start = time.perf_counter()

        try:
            response = query_engine.query(question)
            sources = []

            for source in response.source_nodes:
                sources.append({
                    "file_name": source.node.metadata.get("file_name", "unknown"),
                    "score": float(source.score) if source.score is not None else None,
                    "text": source.node.get_content()[:1000],
                })

            result = {
                "query_number": number,
                "question": question,
                "answer": str(response),
                "latency_seconds": round(time.perf_counter() - query_start, 4),
                "source_nodes": sources,
                "status": "success",
            }

        except Exception as error:
            result = {
                "query_number": number,
                "question": question,
                "answer": "",
                "latency_seconds": round(time.perf_counter() - query_start, 4),
                "source_nodes": [],
                "status": "error",
                "error": str(error),
            }

        results.append(result)
        print("Status:", result["status"])
        print("Latency:", result["latency_seconds"], "seconds")
        print("Answer:", result["answer"][:200])

        if result["status"] == "error":
            print("Error:", result["error"])

    output = {
        "experiment": "LlamaIndex with persistent ChromaDB",
        "embedding_model": "nomic-embed-text",
        "llm_model": "qwen2.5:3b",
        "document_count": len(documents),
        "query_count": len(QUERIES),
        "indexing_seconds": round(indexing_seconds, 4),
        "chroma_collection": "w7d3_llamaindex_documents",
        "chroma_directory": str(CHROMA_DIR),
        "results": results,
    }

    OUTPUT_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    successful = sum(item["status"] == "success" for item in results)
    print(f"\nSuccessful queries: {successful}/10")
    print(f"Indexing time: {indexing_seconds:.4f} seconds")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
    