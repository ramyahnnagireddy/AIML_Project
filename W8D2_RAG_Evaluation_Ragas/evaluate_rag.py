
"""Evaluate W8D2 RAG using Ragas 0.4.3 and local Ollama models."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any

import instructor
from openai import AsyncOpenAI
from rag_pipeline import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    LLM_MODEL,
    RAGPipeline,
)
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATASET_PATH = OUTPUT_DIR / "qa_dataset.json"

EVALUATOR_MODEL = "qwen2.5:3b"

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Save output as readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def calculate_averages(
    rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    """Average successful scores without counting failures as zero."""

    averages: dict[str, float | None] = {}

    for name in METRIC_NAMES:
        scores = [
            row["metrics"][name]
            for row in rows
            if row["metrics"].get(name) is not None
        ]
        averages[name] = (
            round(sum(scores) / len(scores), 4) if scores else None
        )

    return averages


def create_metrics() -> dict[str, Any]:
    """Configure four Ragas metrics with locally hosted models."""

    base_url = "http://localhost:11434/v1"

    raw_chat_client = AsyncOpenAI(
        api_key="ollama",
        base_url=base_url,
        timeout=180.0,
        max_retries=1,
    )

    # Ragas 0.4.3 does not expose the newer llm_factory(mode=...) argument.
    # Apply Markdown-JSON mode directly to the Instructor client instead.
    structured_chat_client = instructor.from_openai(
        raw_chat_client,
        mode=instructor.Mode.MD_JSON,
    )

    evaluator_llm = InstructorLLM(
        client=structured_chat_client,
        model=EVALUATOR_MODEL,
        provider="openai",
       model_args=InstructorModelArgs(
    temperature=0.0,
    top_p=0.9,
    max_tokens=1024,
),
    )

    # Ollama also exposes an OpenAI-compatible embeddings endpoint.
    embedding_client = AsyncOpenAI(
        api_key="ollama",
        base_url=base_url,
        timeout=180.0,
        max_retries=1,
    )

    evaluator_embeddings = OpenAIEmbeddings(
        client=embedding_client,
        model=EMBEDDING_MODEL,
    )

    return {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
    }


async def score_sample(
    metrics: dict[str, Any],
    sample: dict[str, Any],
) -> tuple[dict[str, float | None], dict[str, str]]:
    """Evaluate one answer and record errors separately."""

    question = sample["user_input"]
    reference = sample["reference"]
    response = sample["response"]
    contexts = sample["retrieved_contexts"]

    metric_inputs = {
        "faithfulness": {
            "user_input": question,
            "response": response,
            "retrieved_contexts": contexts,
        },
        "answer_relevancy": {
            "user_input": question,
            "response": response,
        },
        "context_precision": {
            "user_input": question,
            "reference": reference,
            "retrieved_contexts": contexts,
        },
        "context_recall": {
            "user_input": question,
            "reference": reference,
            "retrieved_contexts": contexts,
        },
    }

    scores: dict[str, float | None] = {}
    errors: dict[str, str] = {}

    for name in METRIC_NAMES:
        try:
            result = await metrics[name].ascore(**metric_inputs[name])
            value = float(result.value)

            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Unexpected metric value: {value}")

            scores[name] = value
            print(f"    {name}: {value:.4f}")

        except Exception as exc:  # noqa: BLE001
            scores[name] = None
            message = str(exc).replace("\n", " ")
            errors[name] = f"{type(exc).__name__}: {message[:400]}"
            print(f"    {name} FAILED: {errors[name]}")

    return scores, errors


async def run_evaluation(
    mode: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    limit: int | None,
) -> None:
    """Run baseline or optimised evaluation and persist results."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. "
            "Run generate_dataset.py first."
        )

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    all_questions = dataset.get("samples", [])

    if len(all_questions) != 10:
        raise ValueError(
            f"Expected 10 Q&A pairs, found {len(all_questions)}."
        )

    if limit is not None and limit < 1:
        raise ValueError("--limit must be greater than zero.")

    questions = all_questions[:limit] if limit is not None else all_questions
    is_smoke_test = limit is not None

    if is_smoke_test:
        output_path = OUTPUT_DIR / "evaluation_smoke_test.json"
    else:
        filename = (
            "evaluation_before.json"
            if mode == "before"
            else "evaluation_after.json"
        )
        output_path = OUTPUT_DIR / filename

    print(f"\nW8D2 Ragas evaluation — {mode.upper()}")
    print(f"Questions in this run: {len(questions)}")
    print(f"Full reference dataset: {len(all_questions)}")
    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")
    print(f"Retrieval top-k: {top_k}")
    print(f"RAG generation model: {LLM_MODEL}")
    print(f"Ragas judge model: {EVALUATOR_MODEL}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    print("\nBuilding the RAG index...")
    pipeline = RAGPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
    )

    print("Initialising Ragas evaluators...")
    metrics = create_metrics()
    print("All four metric objects initialised.")

    records: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "mode": mode,
        "smoke_test": is_smoke_test,
        "ragas_version": "0.4.3",
        "sample_count": len(questions),
        "full_dataset_size": len(all_questions),
        "configuration": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            "rag_generation_model": LLM_MODEL,
            "ragas_evaluator_model": EVALUATOR_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "indexed_chunks": pipeline.chunk_count,
        },
        "metric_averages": {name: None for name in METRIC_NAMES},
        "successful_metric_scores": {name: 0 for name in METRIC_NAMES},
        "completed": False,
        "results": records,
    }
    save_json(output_path, payload)

    for index, qa in enumerate(questions, start=1):
        print(f"\n[{index}/{len(questions)}] {qa['user_input']}")

        rag_result = pipeline.query(qa["user_input"])

        record: dict[str, Any] = {
            "id": qa["id"],
            "user_input": qa["user_input"],
            "reference": qa["reference"],
            "response": rag_result["response"],
            "retrieved_contexts": rag_result["retrieved_contexts"],
            "metrics": {name: None for name in METRIC_NAMES},
            "metric_errors": {},
        }

        scores, errors = await score_sample(metrics, record)
        record["metrics"] = scores
        record["metric_errors"] = errors
        records.append(record)

        payload["metric_averages"] = calculate_averages(records)
        payload["successful_metric_scores"] = {
            name: sum(
                row["metrics"].get(name) is not None for row in records
            )
            for name in METRIC_NAMES
        }

        # Persist each completed question as evidence.
        save_json(output_path, payload)

    payload["metric_averages"] = calculate_averages(records)
    payload["successful_metric_scores"] = {
        name: sum(
            row["metrics"].get(name) is not None for row in records
        )
        for name in METRIC_NAMES
    }
    payload["completed"] = True
    save_json(output_path, payload)

    print("\n========== Evaluation Summary ==========")
    for name in METRIC_NAMES:
        score = payload["metric_averages"][name]
        count = payload["successful_metric_scores"][name]

        if score is None:
            print(f"{name}: unavailable ({count}/{len(questions)} scores)")
        else:
            print(f"{name}: {score:.4f} ({count}/{len(questions)} scores)")

    print(f"\nResults saved to: {output_path}")

    if any(row["metric_errors"] for row in records):
        print("Warning: some metrics failed. Inspect metric_errors in JSON.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate W8D2 RAG using Ragas 0.4.3."
    )
    parser.add_argument(
        "--mode",
        choices=["before", "after"],
        default="before",
    )
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N samples for a smoke test.",
    )
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            mode=args.mode,
            chunk_size=(
                args.chunk_size
                if args.chunk_size is not None
                else DEFAULT_CHUNK_SIZE
            ),
            chunk_overlap=(
                args.chunk_overlap
                if args.chunk_overlap is not None
                else DEFAULT_CHUNK_OVERLAP
            ),
            top_k=(
                args.top_k if args.top_k is not None else DEFAULT_TOP_K
            ),
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
    