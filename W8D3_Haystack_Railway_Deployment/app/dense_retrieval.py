

from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_pipeline import load_pdf_documents

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_dense_pipeline():
    documents = load_pdf_documents()

    document_embedder = SentenceTransformersDocumentEmbedder(
        model=EMBEDDING_MODEL
    )
    document_embedder.warm_up()

    embedded_result = document_embedder.run(documents=documents)
    embedded_documents = embedded_result["documents"]

    document_store = InMemoryDocumentStore(
        embedding_similarity_function="cosine"
    )
    document_store.write_documents(embedded_documents)

    text_embedder = SentenceTransformersTextEmbedder(
        model=EMBEDDING_MODEL
    )
    text_embedder.warm_up()

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component(
        "retriever",
        InMemoryEmbeddingRetriever(document_store=document_store),
    )
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    return pipeline, document_store


def retrieve_dense(pipeline, question, top_k=3):
    result = pipeline.run(
        {
            "text_embedder": {"text": question},
            "retriever": {"top_k": top_k},
        }
    )

    return [
        {
            "source": document.meta.get("source", "unknown"),
            "score": float(document.score or 0.0),
            "content": document.content or "",
        }
        for document in result["retriever"]["documents"]
    ]
