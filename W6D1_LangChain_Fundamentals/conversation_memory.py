"""
W6D1 - LangChain Fundamentals
Task 2: Conversation Memory

Maintains conversation history across 5 turns using LangChain
message history and ChatOllama.
"""

from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama


MODEL_NAME = "llama3.2:3b"


def main():
    """Run a five-turn conversation while maintaining history."""

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    conversation_history = []

    turns = [
        "My name is Ramya.",
        "I am learning machine learning.",
        "What is a good beginner topic for me to study?",
        "Can you remind me what I said I am learning?",
        "What should I focus on next?"
    ]

    print("=" * 70)
    print("W6D1 - Conversation Memory Test")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print("Conversation turns: 5")
    print("=" * 70)

    for turn_number, user_input in enumerate(turns, start=1):
        print(f"\nTURN {turn_number}")
        print(f"USER: {user_input}")
        print("-" * 70)

        conversation_history.append(
            HumanMessage(content=user_input)
        )

        system_instruction = """
You are a helpful AI/ML mentor for a beginner.

Use the conversation history to answer the user's current question.
Remember important information stated earlier in the conversation.
Keep the answer clear and beginner-friendly.
"""

        messages = [
            (
                "system",
                system_instruction
            )
        ] + conversation_history

        try:
            response = llm.invoke(messages)

            conversation_history.append(
                AIMessage(content=response.content)
            )

            print("ASSISTANT:")
            print(response.content)

            print(f"\nHistory messages stored: {len(conversation_history)}")

        except Exception as error:
            print(f"ERROR: {error}")


if __name__ == "__main__":
    main()
    