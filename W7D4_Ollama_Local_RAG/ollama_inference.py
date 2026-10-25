import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# Directory configuration
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ollama configuration
OLLAMA_API = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

# Custom system prompt
SYSTEM_PROMPT = """
You are a helpful AI/ML learning assistant.
Provide accurate, beginner-friendly explanations.
Use clear steps for technical questions.
For coding questions, provide simple examples.
If uncertain, say so instead of inventing facts.
Keep answers focused and easy to understand.
"""

# Five test prompts
TEST_PROMPTS = [
    "What is artificial intelligence? Explain with two examples.",
    "Explain supervised learning in simple words.",
    "What is the difference between classification and regression?",
    "Write a short Python function to calculate a factorial.",
    "Why is train-test splitting important in machine learning?",
]


def call_ollama(prompt: str) -> dict:
    """Send a prompt to the local Ollama chat API."""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": prompt,
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
            "model": MODEL,
            "prompt": prompt,
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
            "model": MODEL,
            "prompt": prompt,
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

    print("=" * 60)
    print("W7D4: OLLAMA LOCAL LLM INFERENCE")
    print(f"Model: {MODEL}")
    print("=" * 60)

    for index, prompt in enumerate(TEST_PROMPTS, start=1):
        print(f"\nTest {index}/{len(TEST_PROMPTS)}")
        print(f"Prompt: {prompt}")

        result = call_ollama(prompt)
        results.append(result)

        print(f"Status: {result['status']}")
        print(f"Latency: {result['latency_seconds']} seconds")

        if result["status"] == "success":
            print(f"Response:\n{result['response']}")
        else:
            print(f"Error: {result['error']}")

    successful_count = sum(
        result["status"] == "success"
        for result in results
    )

    output = {
        "experiment": "W7D4 Ollama Local LLM Inference",
        "model": MODEL,
        "system_prompt": SYSTEM_PROMPT.strip(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_prompts": len(TEST_PROMPTS),
        "successful_prompts": successful_count,
        "results": results,
    }

    output_path = OUTPUT_DIR / "inference_results.json"

    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"Successful tests: {successful_count}/{len(TEST_PROMPTS)}")
    print(f"Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
    