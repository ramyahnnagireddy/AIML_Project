
"""MLflow tracking and bounded Ragas faithfulness evaluation."""

from __future__ import annotations

import math
from typing import Any

EXPERIMENT_NAME = "W9D5_Automated_Research_Report_Agent"
OLLAMA_BASE_URL = "http://localhost:11434"
EVALUATOR_MODEL = "qwen2.5:3b"
MAX_EVALUATION_ANSWER_CHARS = 2000
MAX_CONTEXT_CHARS = 1000
MAX_EVALUATION_CONTEXTS = 2
EVALUATION_TIMEOUT_SECONDS = 120


def log_research_run(
    topic: str,
    source_count: int,
    report_length: int,
    status: str,
) -> str | None:
    """Log the research pipeline run to MLflow."""
    try:
        import mlflow
    except ImportError:
        print("Warning: MLflow is not installed; tracking was skipped.")
        return None

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_param("topic", topic[:250])
        mlflow.log_param("source_count", source_count)
        mlflow.log_param("status", status)
        mlflow.log_metric("report_characters", report_length)
        mlflow.set_tag("project_day", "W9D5")
        return run.info.run_id


def evaluate_with_ragas(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, Any]:
    """Evaluate faithfulness using a local Ollama model.

    Inputs and model output are bounded. Failed evaluations are recorded
    honestly and never produce a fabricated score.
    """
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if not answer.strip():
        raise ValueError("Answer must not be empty.")

    clean_contexts = [
        context.strip()[:MAX_CONTEXT_CHARS]
        for context in contexts
        if isinstance(context, str) and context.strip()
    ][:MAX_EVALUATION_CONTEXTS]

    result: dict[str, Any] = {
        "evaluation_status": "not_evaluated_no_contexts",
        "metric": "faithfulness",
        "scores": {},
        "contexts_used": len(clean_contexts),
        "answer_truncated": len(answer) > MAX_EVALUATION_ANSWER_CHARS,
        "note": (
            "No usable retrieved contexts were supplied; "
            "Ragas evaluation was not run."
        ),
    }

    if not clean_contexts:
        return result

    try:
        from datasets import Dataset
        from langchain_ollama import ChatOllama
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness
        from ragas.run_config import RunConfig

        judge_model = ChatOllama(
            model=EVALUATOR_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            num_predict=128,
            timeout=EVALUATION_TIMEOUT_SECONDS,
        )
        judge = LangchainLLMWrapper(judge_model)
        faithfulness = Faithfulness(llm=judge)

        dataset = Dataset.from_dict(
            {
                "user_input": [question[:500]],
                "response": [
    answer.split("## Executive Summary", 1)[-1]
    .split("\n## ", 1)[0]
    .strip()[:MAX_EVALUATION_ANSWER_CHARS]
],
                "retrieved_contexts": [clean_contexts],
            }
        )

        run_config = RunConfig(
            timeout=EVALUATION_TIMEOUT_SECONDS,
            max_retries=0,
            max_workers=1,
        )

        evaluation = evaluate(
            dataset=dataset,
            metrics=[faithfulness],
            run_config=run_config,
            raise_exceptions=False,
            show_progress=False,
        )

        scores_frame = evaluation.to_pandas()
        if scores_frame.empty:
            raise RuntimeError("Ragas returned no evaluation rows.")

        raw_score = scores_frame.iloc[0].get("faithfulness")
        if raw_score is None:
            raise RuntimeError("Ragas did not return a faithfulness score.")

        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError("Ragas returned an invalid faithfulness score.")

        result.update(
            {
                "evaluation_status": "completed",
                "scores": {"faithfulness": score},
                "note": (
                    f"Evaluated using {EVALUATOR_MODEL}. The response is "
                    f"limited to {MAX_EVALUATION_ANSWER_CHARS} characters "
                    f"and each context to {MAX_CONTEXT_CHARS} characters. "
                    "The score measures support from supplied snippets, "
                    "not overall factual correctness."
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "evaluation_status": "failed",
                "scores": {},
                "error": f"{type(exc).__name__}: {exc}",
                "note": (
                    "Ragas did not produce a valid faithfulness score. "
                    "Inspect the error and local model availability."
                ),
            }
        )

    return result
