from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


def main():
    llm = OllamaLLM(model="llama3.2:3b")

    prompt = PromptTemplate(
        input_variables=["history", "question"],
        template=(
            "You are an AI/ML mentor for a beginner.\n\n"
            "Conversation history:\n"
            "{history}\n\n"
            "Current question:\n"
            "{question}\n\n"
            "Answer the current question clearly and briefly. "
            "Use the conversation history when relevant."
        ),
    )

    chain = prompt | llm

    questions = [
        "What is machine learning?",
        "What are its three common types?",
        "Which type uses labeled data?",
        "Can you give me one example of it?",
        "Why is it useful in real-world applications?",
    ]

    conversation_history = []

    output_lines = [
        "W6D3 TASK 2 - CONVERSATION MEMORY",
        "=" * 60,
        "Model: llama3.2:3b",
        "Memory approach: accumulated conversation history",
        "Turns tested: 5",
        "",
    ]

    for turn, question in enumerate(questions, start=1):
        history_text = "\n".join(conversation_history)

        response = chain.invoke(
            {
                "history": history_text,
                "question": question,
            }
        )

        conversation_history.append(f"User: {question}")
        conversation_history.append(f"Assistant: {response}")

        print(f"\nTURN {turn}")
        print(f"User: {question}")
        print(f"Assistant: {response}")

        output_lines.append(f"TURN {turn}")
        output_lines.append(f"USER: {question}")
        output_lines.append("ASSISTANT:")
        output_lines.append(response)
        output_lines.append("")
        output_lines.append("MEMORY AFTER TURN:")
        output_lines.extend(conversation_history)
        output_lines.append("-" * 60)

    output_lines.append("")
    output_lines.append("FINAL CONVERSATION HISTORY")
    output_lines.append("=" * 60)
    output_lines.extend(conversation_history)

    with open(
        "W6D3_LangChain_Tools_Agents/outputs/memory_output.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(output_lines))

    print("\nConversation memory test completed.")
    print("5 turns completed successfully.")
    print("Conversation history was maintained across all turns.")
    print("Evidence saved to outputs/memory_output.txt")


if __name__ == "__main__":
    main()
    