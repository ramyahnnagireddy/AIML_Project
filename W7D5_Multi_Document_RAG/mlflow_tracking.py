
"""MLflow experiment tracking for the W7D5 RAG system."""

import json

import mlflow

from config import MLFLOW_EXPERIMENT_NAME, OUTPUTS_DIR


def start_tracking() -> None:
    """Configure MLflow to use a local SQLite tracking database."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    database_path = (OUTPUTS_DIR / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{database_path.as_posix()}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_rag_run(
    question: str,
    answer: str,
    retrieved_chunks: list[dict],
    latency_seconds: float,
    answer_source: str = "langgraph",
) -> str:
    """Record a RAG query, its metrics, and its output artifact."""
    start_tracking()

    with mlflow.start_run() as run:
        mlflow.log_param("answer_source", answer_source)
        mlflow.log_param("top_k", len(retrieved_chunks))
        mlflow.log_param("question", question[:250])

        mlflow.log_metric("latency_seconds", float(latency_seconds))
        mlflow.log_metric("retrieved_chunk_count", len(retrieved_chunks))

        mean_score = (
            sum(chunk["score"] for chunk in retrieved_chunks)
            / len(retrieved_chunks)
            if retrieved_chunks
            else 0.0
        )
        mlflow.log_metric("mean_retrieval_score", mean_score)

        artifact_path = OUTPUTS_DIR / "latest_mlflow_result.json"
        artifact_path.write_text(
            json.dumps(
                {
                    "question": question,
                    "answer": answer,
                    "retrieved_chunks": retrieved_chunks,
                    "latency_seconds": latency_seconds,
                    "answer_source": answer_source,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(artifact_path), artifact_path="rag_output")
        return run.info.run_id


if __name__ == "__main__":
    run_id = log_rag_run(
        question="What is RAG?",
        answer="RAG combines retrieval with language generation.",
        retrieved_chunks=[
            {
                "source": "rag_fundamentals.txt",
                "text": "RAG combines document retrieval with a language model.",
                "score": 0.85,
            }
        ],
        latency_seconds=1.25,
    )

    print("MLflow tracking test succeeded.")
    print("Run ID:", run_id)
    print("SQLite database:", OUTPUTS_DIR / "mlflow.db")
    