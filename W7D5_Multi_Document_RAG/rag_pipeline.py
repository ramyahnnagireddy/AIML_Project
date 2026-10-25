
"""Multi-document RAG pipeline orchestrated with LangGraph and MLflow."""

import json
import time
from datetime import datetime, timezone
from typing import TypedDict

import requests
from langgraph.graph import END, START, StateGraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    DOCUMENTS_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OUTPUTS_DIR,
    TOP_K,
)
from mlflow_tracking import log_rag_run


class RAGState(TypedDict, total=False):
    question: str
    retrieved_chunks: list[dict]
    answer: str
    error: str


def load_documents() -> list[dict]:
    """Load text documents and group sentences into chunks."""
    documents = []

    for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        sentences = [
            sentence.strip()
            for sentence in text.replace("\n", " ").split(".")
            if sentence.strip()
        ]

        for index in range(0, len(sentences), 2):
            chunk_text = ". ".join(sentences[index:index + 2]) + "."
            documents.append({
                "source": path.name,
                "chunk_id": index // 2,
                "text": chunk_text,
            })

    if not documents:
        raise FileNotFoundError(
            f"No non-empty text documents found in {DOCUMENTS_DIR}"
        )

    return documents


def retrieve_chunks(question: str, documents: list[dict]) -> list[dict]:
    """Retrieve the most relevant chunks using TF-IDF cosine similarity."""
    texts = [doc["text"] for doc in documents]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()

    indices = scores.argsort()[::-1][:TOP_K]
    return [
        {**documents[i], "score": round(float(scores[i]), 4)}
        for i in indices
        if scores[i] > 0
    ]


def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant context for a question."""
    question = state.get("question", "").strip()
    if not question:
        return {"retrieved_chunks": [], "error": "Question cannot be empty."}

    return {"retrieved_chunks": retrieve_chunks(question, load_documents())}


def generate_node(state: RAGState) -> RAGState:
    """Generate an answer from retrieved context with local Ollama."""
    if state.get("error"):
        return {"answer": "Unable to answer: " + state["error"]}

    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {
            "answer": (
                "No relevant information was found in the supplied documents."
            )
        }

    context = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}" for item in chunks
    )

    prompt = f"""Answer using only the context below.
If the answer is not present, say that the documents do not provide it.
Mention the relevant source filenames and do not invent facts.

Context:
{context}

Question: {state['question']}

Answer:"""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    answer = response.json().get("response", "").strip()

    return {"answer": answer or "The model returned an empty answer."}


def build_graph():
    """Compile the retrieval-and-generation LangGraph."""
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def ask_question(question: str) -> dict:
    """Run RAG, save evidence, and log the query to MLflow."""
    start = time.perf_counter()
    result = build_graph().invoke({"question": question})
    latency = round(time.perf_counter() - start, 3)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "answer": result.get("answer", ""),
        "error": result.get("error"),
        "latency_seconds": latency,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUTS_DIR / "rag_results.json"

    existing = []
    if output_file.exists():
        try:
            existing = json.loads(output_file.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []

    try:
        run_id = log_rag_run(
            question=question,
            answer=record["answer"],
            retrieved_chunks=record["retrieved_chunks"],
            latency_seconds=latency,
            answer_source="langgraph_ollama",
        )
        record["mlflow_run_id"] = run_id
        print(f"MLflow run: {run_id}")
    except Exception as exc:
        print(f"Warning: MLflow logging failed: {exc}")

    existing.append(record)
    output_file.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return record


if __name__ == "__main__":
    print("Multi-document RAG — LangGraph + MLflow")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            print("Please enter a question.")
            continue

        try:
            result = ask_question(question)

            print("\nRetrieved sources:")
            for chunk in result["retrieved_chunks"]:
                print(f"- {chunk['source']} (score={chunk['score']})")

            print("\nAnswer:", result["answer"])
            print("Latency:", result["latency_seconds"], "seconds")
            print("Evidence saved to outputs/rag_results.json")
        except requests.RequestException as exc:
            print("Could not reach Ollama. Check that Ollama is running.")
            print("Details:", exc)
        except Exception as exc:
            print(f"Pipeline error: {exc}")
            