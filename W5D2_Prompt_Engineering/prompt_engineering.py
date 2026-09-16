"""
W5D2: Prompt Engineering & System Prompts with Ollama

This script demonstrates:
1. Calling a local Ollama model through the Python API
2. Using a custom system prompt
3. Testing five different prompts
4. Saving output evidence
"""

from pathlib import Path

import ollama


MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """
You are an AI/ML mentor for engineering students who are beginners.

Your responsibilities:
- Explain concepts clearly and simply.
- Use practical examples when useful.
- Avoid unnecessary jargon.
- Break difficult topics into small steps.
- Give accurate and concise answers.
"""


PROMPTS = [
    "What is machine learning?",
    "Explain supervised learning with a simple example.",
    "What is the difference between classification and regression?",
    "Why do we split a dataset into training and testing sets?",
    "What is overfitting and how can it be reduced?",
]


def ask_model(prompt: str) -> str:
    """Send a prompt to the local Ollama model."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response["message"]["content"]


def main() -> None:
    """Run all prompt tests and save the results."""
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "prompt_test_output.txt"

    with output_file.open("w", encoding="utf-8") as file:
        file.write("W5D2 - Prompt Engineering with Ollama\n")
        file.write("=" * 60 + "\n")
        file.write(f"Model: {MODEL}\n")
        file.write("System Prompt: AI/ML mentor for beginners\n")
        file.write("=" * 60 + "\n\n")

        for index, prompt in enumerate(PROMPTS, start=1):
            print(f"\nPrompt {index}: {prompt}")

            answer = ask_model(prompt)

            print("Response:")
            print(answer)

            file.write(f"PROMPT {index}\n")
            file.write("-" * 60 + "\n")
            file.write(f"{prompt}\n\n")
            file.write("RESPONSE\n")
            file.write("-" * 60 + "\n")
            file.write(f"{answer}\n\n")

    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
    