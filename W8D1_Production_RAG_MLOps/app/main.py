
"""Production-style FastAPI wrapper for the W7D5 RAG pipeline."""

import logging
import sys
import time
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("w8d1.api")

# Support running from the repository root and inside the Docker image.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAG_PROJECT = REPOSITORY_ROOT / "W7D5_Multi_Document_RAG"

if str(RAG_PROJECT) not in sys.path:
    sys.path.insert(0, str(RAG_PROJECT))

import rag_pipeline

app = FastAPI(
    title="Production RAG API",
    description="FastAPI service for the W7D5 multi-document RAG pipeline.",
    version="1.0.0",
)

REQUEST_COUNT = 0
ERROR_COUNT = 0


class QueryRequest(BaseModel):
    question: Annotated[
        str,
        Field(min_length=1, max_length=2000, description="Question for the RAG system"),
    ]


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    latency_seconds: float


@app.middleware("http")
async def record_request_metrics(request: Request, call_next):
    """Record request counts, errors, and latency in application logs."""
    global REQUEST_COUNT, ERROR_COUNT

    started = time.perf_counter()
    REQUEST_COUNT += 1
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        if status_code >= 500:
            ERROR_COUNT += 1
        return response
    except Exception:
        ERROR_COUNT += 1
        raise
    finally:
        latency = time.perf_counter() - started
        logger.info(
            "request method=%s path=%s status=%s latency_seconds=%.4f "
            "request_count=%s error_count=%s",
            request.method,
            request.url.path,
            status_code,
            latency,
            REQUEST_COUNT,
            ERROR_COUNT,
        )


@app.get("/")
def root():
    """Describe the API."""
    return {
        "service": "Production RAG API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    """Check API process health without requiring Ollama."""
    return {
        "status": "healthy",
        "service": "production-rag-api",
        "ollama_configured_url": rag_pipeline.OLLAMA_BASE_URL,
    }


@app.get("/metrics")
def metrics():
    """Expose basic process-local counters for demonstration."""
    return {
        "requests_total": REQUEST_COUNT,
        "server_errors_total": ERROR_COUNT,
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    """Run the existing RAG pipeline for a user question."""
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty or whitespace.",
        )

    try:
        result = rag_pipeline.ask_question(question)
    except Exception:
        logger.exception("RAG query failed")
        raise HTTPException(
            status_code=502,
            detail="RAG processing failed. Check the API and Ollama logs.",
        ) from None

    chunks = result.get("retrieved_chunks", [])
    sources = sorted(
        {chunk["source"] for chunk in chunks if chunk.get("source")}
    )

    return QueryResponse(
        question=question,
        answer=result.get("answer", ""),
        sources=sources,
        latency_seconds=float(result.get("latency_seconds", 0.0)),
    )
