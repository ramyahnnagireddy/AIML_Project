
"""Local RAG pipeline for W8D2 Ragas evaluation."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "knowledge_base.txt"

EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "qwen2.5:3b"

DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 2


def split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text into overlapping chunks without extra ML dependencies."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        end = hard_end

        # Prefer natural boundaries when one is available near the end
        # of the chunk. Otherwise, use the character limit.
        if hard_end < text_length:
            minimum_boundary = start + int(chunk_size * 0.55)

            for separator in ("\n\n", "\n", ". ", " "):
                boundary = text.rfind(
                    separator,
                    minimum_boundary,
                    hard_end,
                )
                if boundary >= minimum_boundary:
                    end = boundary + len(separator)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Advance while keeping the requested amount of overlap.
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + 1

        start = next_start

    return chunks


class RAGPipeline:
    """Retrieve relevant context from ChromaDB and generate answers."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size."
            )
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: {DATA_PATH}"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        text = DATA_PATH.read_text(encoding="utf-8")
        chunks = split_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not chunks:
            raise ValueError("Knowledge base contains no usable text.")

        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": DATA_PATH.name,
                    "chunk_id": index,
                },
            )
            for index, chunk in enumerate(chunks)
        ]

        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        # A unique in-memory collection prevents generated database files
        # from accumulating in the repository.
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=f"w8d2_{uuid4().hex[:12]}",
        )

        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )

        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant answering questions from a knowledge base.

Use only information in the retrieved context.
Do not invent facts or add unsupported outside information.
If the context does not contain enough evidence, clearly acknowledge
that limitation.

Retrieved context:
{context}

Question:
{question}

Answer:"""
        )

        model = ChatOllama(
            model=LLM_MODEL,
            temperature=0,
        )
        self.answer_chain = prompt | model | StrOutputParser()
        self.chunk_count = len(documents)

    def query(self, question: str) -> dict[str, object]:
        """Return a generated response and its retrieved contexts."""

        if not question.strip():
            raise ValueError("Question must not be empty.")

        retrieved_documents = self.retriever.invoke(question)
        contexts = [
            document.page_content
            for document in retrieved_documents
        ]

        answer = self.answer_chain.invoke(
            {
                "question": question,
                "context": "\n\n".join(contexts),
            }
        )

        return {
            "question": question,
            "response": answer.strip(),
            "retrieved_contexts": contexts,
        }


def main() -> None:
    """Run a small RAG pipeline verification."""

    print("W8D2 — RAG Pipeline Smoke Test")

    pipeline = RAGPipeline()
    result = pipeline.query(
        "What is Retrieval-Augmented Generation?"
    )

    print(f"\nChunk size: {pipeline.chunk_size}")
    print(f"Chunk overlap: {pipeline.chunk_overlap}")
    print(f"Retrieval top-k: {pipeline.top_k}")
    print(f"Indexed chunks: {pipeline.chunk_count}")
    print(f"Retrieved contexts: {len(result['retrieved_contexts'])}")
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['response']}")
    print("\nSmoke test completed.")


if __name__ == "__main__":
    main()
    