import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# Directory configuration
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ollama API configuration
OLLAMA_API = "http://localhost:11434/api/chat"

MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b",
]

# Use the same system prompt for both models
SYSTEM_PROMPT = """
You are a helpful AI/ML learning assistant.
Provide accurate, beginner-friendly explanations.
Keep answers clear, focused, and technically correct.
Use examples when they help explain a concept.
If uncertain, acknowledge the uncertainty.
"""

# Three identical questions for both models
TEST_QUESTIONS = [
    "Explain supervised learning with one simple example.",
    "What is the difference between classification and regression?",
    "Explain three important steps in a machine learning project.",
]


def call_ollama(model: str, question: str) -> dict:
    """Send a question to a selected local Ollama model."""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    start_time = time.perf_counter()

    try:
        response = requests.post(
            OLLAMA_API,
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        return {
            "status": "success",
            "model": model,
            "question": question,
            "response": data.get("message", {}).get("content", ""),
            "latency_seconds": round(
                time.perf_counter() - start_time, 3
            ),
            "prompt_tokens": data.get("prompt_eval_count"),
            "response_tokens": data.get("eval_count"),
            "error": None,
        }

    except (requests.RequestException, ValueError) as exc:
        return {
            "status": "error",
            "model": model,
            "question": question,
            "response": "",
            "latency_seconds": round(
                time.perf_counter() - start_time, 3
            ),
            "prompt_tokens": None,
            "response_tokens": None,
            "error": str(exc),
        }


def main() -> None:
    results = []

    print("=" * 70)
    print("W7D4: LOCAL OLLAMA MODEL COMPARISON")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Questions per model: {len(TEST_QUESTIONS)}")
    print("=" * 70)

    for question_number, question in enumerate(
        TEST_QUESTIONS, start=1
    ):
        print(f"\nQuestion {question_number}/{len(TEST_QUESTIONS)}")
        print(f"Question: {question}")

        for model in MODELS:
            print(f"\nTesting model: {model}")

            result = call_ollama(model, question)
            results.append(result)

            print(f"Status: {result['status']}")
            print(f"Latency: {result['latency_seconds']} seconds")

            if result["status"] == "success":
                print(f"Response:\n{result['response']}")
            else:
                print(f"Error: {result['error']}")

    # Calculate measured performance statistics
    model_summary = {}

    for model in MODELS:
        model_results = [
            item for item in results
            if item["model"] == model
        ]

        successful = [
            item for item in model_results
            if item["status"] == "success"
        ]

        average_latency = (
            round(
                sum(item["latency_seconds"] for item in successful)
                / len(successful),
                3,
            )
            if successful
            else None
        )

        model_summary[model] = {
            "total_questions": len(TEST_QUESTIONS),
            "successful_responses": len(successful),
            "failed_responses": len(model_results) - len(successful),
            "average_latency_seconds": average_latency,
        }

    output = {
        "experiment": "W7D4 Ollama Model Comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_endpoint": OLLAMA_API,
        "system_prompt": SYSTEM_PROMPT.strip(),
        "temperature": 0.2,
        "models": MODELS,
        "questions": TEST_QUESTIONS,
        "model_summary": model_summary,
        "results": results,
    }

    output_path = OUTPUT_DIR / "model_comparison.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")

    for model, summary in model_summary.items():
        print(f"\nModel: {model}")
        print(
            f"Successful responses: "
            f"{summary['successful_responses']}/"
            f"{summary['total_questions']}"
        )
        print(
            f"Average latency: "
            f"{summary['average_latency_seconds']} seconds"
        )

    print(f"\nResults saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    