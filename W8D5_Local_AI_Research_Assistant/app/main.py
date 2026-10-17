"""FastAPI application for the Local AI Research Assistant."""

import os
from pathlib import Path

import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from research_assistant import ResearchAssistant
from workflow import build_research_graph

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use SQLite instead of MLflow's deprecated file-based tracking store.
MLFLOW_DB = OUTPUT_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
mlflow.set_experiment("W8D5_Local_AI_Research_Assistant")

assistant = ResearchAssistant(
    use_ollama=os.getenv("USE_OLLAMA", "false").lower() == "true",
    model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
)

research_graph = build_research_graph(assistant)

app = FastAPI(
    title="Local AI Research Assistant",
    description="Ask questions grounded in your local research notes.",
    version="1.0.0",
)


class ResearchRequest(BaseModel):
    """Validated research request."""

    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only questions."""
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("Question cannot be blank.")

        return cleaned


@app.get("/health")
def health() -> dict:
    """Report that the API process is running."""
    return {
        "status": "healthy",
        "service": "local-ai-research-assistant",
    }


@app.post("/research")
def research(request: ResearchRequest) -> dict:
    """Run the research workflow and record basic experiment metrics."""
    try:
        result = research_graph.invoke(
            {"question": request.question}
        )

        sources = result.get("sources", [])[: request.top_k]

        with mlflow.start_run(run_name="research_query"):
            mlflow.log_param(
                "model",
                result.get("model", "unknown"),
            )
            mlflow.log_param("top_k", request.top_k)
            mlflow.log_metric(
                "retrieved_source_count",
                len(sources),
            )

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": sources,
            "model": result.get("model", "unknown"),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Research request failed: {exc}",
        ) from exc
    