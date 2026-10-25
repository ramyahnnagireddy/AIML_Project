"""
W5D2: Ollama Model Comparison

Compares llama3.2:3b and qwen2.5:3b
using the same three prompts.
"""

from pathlib import Path

import ollama


MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b",
]

PROMPTS = [
    "What is machine learning?",
    "Explain supervised learning with a simple example.",
    "What is overfitting and how can it be reduced?",
]


def ask_model(model: str, prompt: str) -> str:
    """Send a prompt to an Ollama model and return its response."""
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI/ML mentor for engineering students "
                    "who are beginners. Explain concepts clearly and "
                    "simply, using practical examples when useful."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response["message"]["content"]


def main() -> None:
    """Compare both models using the same prompts."""
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "model_comparison.txt"

    with output_file.open("w", encoding="utf-8") as file:
        file.write("W5D2 - Ollama Model Comparison\n")
        file.write("=" * 70 + "\n")
        file.write("Models: llama3.2:3b vs qwen2.5:3b\n")
        file.write("Same system prompt and same three questions used.\n")
        file.write("=" * 70 + "\n\n")

        for prompt_number, prompt in enumerate(PROMPTS, start=1):
            file.write(f"QUESTION {prompt_number}\n")
            file.write("-" * 70 + "\n")
            file.write(f"{prompt}\n\n")

            for model in MODELS:
                print(f"\nModel: {model}")
                print(f"Prompt: {prompt}")

                answer = ask_model(model, prompt)

                print("Response:")
                print(answer)

                file.write(f"MODEL: {model}\n")
                file.write("-" * 70 + "\n")
                file.write(f"{answer}\n\n")

            file.write("=" * 70 + "\n\n")

    print(f"\nComparison output saved to: {output_file}")


if __name__ == "__main__":
    main()
    