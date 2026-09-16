import ollama
from pathlib import Path

MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """
You are a helpful AI/ML mentor.
Explain technical concepts clearly and simply for a beginner.
Keep your answers accurate, concise, and easy to understand.
Use examples when they help.
"""

PROMPTS = [
    "What is machine learning?",
    "Explain supervised learning with a simple example.",
    "What is the difference between classification and regression?",
    "Why do we split data into training and testing sets?",
    "What is overfitting in machine learning?"
]


def run_inference(prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response["message"]["content"]


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "llama3_inference_output.txt"

    lines = [
        "W5D1 - Llama 3.2 Local Inference Evidence",
        f"Model: {MODEL}",
        "Inference method: Ollama Python API",
        "System prompt: AI/ML mentor for beginners",
        "Number of prompts tested: 5",
        ""
    ]

    print("=" * 70)
    print("W5D1 - LOCAL LLM INFERENCE USING OLLAMA")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print("System prompt: AI/ML mentor for beginners")
    print()

    for index, prompt in enumerate(PROMPTS, start=1):
        response = run_inference(prompt)

        print("-" * 70)
        print(f"PROMPT {index}")
        print("-" * 70)
        print(f"Question: {prompt}")
        print()
        print("Response:")
        print(response)
        print()

        lines.extend([
            "-" * 70,
            f"PROMPT {index}",
            "-" * 70,
            f"Question: {prompt}",
            "",
            "Response:",
            response,
            ""
        ])

    output_file.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 70)
    print(f"Evidence saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    