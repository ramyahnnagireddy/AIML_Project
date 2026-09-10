import ollama
from pathlib import Path

MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

SYSTEM_PROMPT = """
You are a helpful AI/ML mentor.
Explain technical concepts clearly and simply for a beginner.
Keep your answers accurate, concise, and easy to understand.
Use examples when they help.
"""

QUESTIONS = [
    "What is supervised learning? Give one simple example.",
    "Explain the difference between classification and regression.",
    "What is overfitting and how can it be reduced?"
]


def get_response(model, question):
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 200
        }
    )

    return response["message"]["content"]


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "model_comparison.txt"

    lines = [
        "W5D1 - Llama 3.2 vs Qwen 2.5 Comparison",
        "",
        "Models tested:",
        "- llama3.2:3b",
        "- qwen2.5:3b",
        "",
        "System prompt: AI/ML mentor for beginners",
        "Number of questions: 3",
        ""
    ]

    print("=" * 70)
    print("W5D1 - LLAMA 3.2 vs QWEN 2.5 MODEL COMPARISON")
    print("=" * 70)

    for question_number, question in enumerate(QUESTIONS, start=1):
        print()
        print("-" * 70)
        print(f"QUESTION {question_number}")
        print("-" * 70)
        print(f"Question: {question}")

        lines.extend([
            "-" * 70,
            f"QUESTION {question_number}",
            "-" * 70,
            f"Question: {question}",
            ""
        ])

        for model in MODELS:
            print()
            print(f"[{model}]")
            print("Generating response...")

            try:
                response = get_response(model, question)

                print(response)

                lines.extend([
                    f"[{model}]",
                    response,
                    ""
                ])

            except Exception as error:
                error_message = f"ERROR: {error}"
                print(error_message)

                lines.extend([
                    f"[{model}]",
                    error_message,
                    ""
                ])

        output_file.write_text(
            "\n".join(lines),
            encoding="utf-8"
        )

    print()
    print("=" * 70)
    print(f"Comparison evidence saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    