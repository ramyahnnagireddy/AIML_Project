

from contextlib import asynccontextmanager

from dense_retrieval import build_dense_pipeline, retrieve_dense
from fastapi import FastAPI, HTTPException
from haystack_pipeline import build_bm25_pipeline, retrieve_bm25
from pydantic import BaseModel, Field

pipelines = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    bm25_pipeline, _ = build_bm25_pipeline()
    pipelines["bm25"] = bm25_pipeline

    dense_pipeline, _ = build_dense_pipeline()
    pipelines["dense"] = dense_pipeline

    yield
    pipelines.clear()


app = FastAPI(
    title="Haystack Retrieval API",
    description="Compare BM25 and dense retrieval over five PDF documents.",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    retriever: str = Field(default="dense")
    top_k: int = Field(default=3, ge=1, le=5)


@app.get("/")
def root():
    return {
        "message": "Haystack Retrieval API",
        "endpoints": ["/health", "/query", "/docs"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "initialized": sorted(pipelines.keys()),
    }


@app.post("/query")
def query_documents(request: QueryRequest):
    retriever_name = request.retriever.lower().strip()

    if retriever_name not in ("bm25", "dense"):
        raise HTTPException(
            status_code=400,
            detail="retriever must be 'bm25' or 'dense'",
        )

    if retriever_name not in pipelines:
        raise HTTPException(
            status_code=503,
            detail="Retrieval pipeline is not initialized",
        )

    if retriever_name == "bm25":
        documents = retrieve_bm25(
            pipelines["bm25"],
            request.question,
            top_k=request.top_k,
        )
    else:
        documents = retrieve_dense(
            pipelines["dense"],
            request.question,
            top_k=request.top_k,
        )

    return {
        "question": request.question,
        "retriever": retriever_name,
        "results": [
            {
                "source": document["source"],
                "score": document["score"],
                "content": document["content"],
            }
            for document in documents
        ],
    }
