from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_classic.memory import ConversationBufferMemory


MODEL_NAME = "llama3.2:3b"


def build_chain():
    """Build a PromptTemplate -> Ollama LLM -> OutputParser chain."""
    prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are an AI/ML mentor for beginners.\n"
            "Answer the following question clearly and briefly.\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
    )

    llm = OllamaLLM(model=MODEL_NAME)

    parser = StrOutputParser()

    chain = prompt | llm | parser

    return chain


def test_chain():
    """Test the LangChain chain with five inputs."""
    chain = build_chain()

    questions = [
        "What is machine learning?",
        "What is supervised learning?",
        "What is a dataset?",
        "What is overfitting?",
        "What is the purpose of train-test split?",
    ]

    results = []

    print("=" * 70)
    print("TASK 1: LANGCHAIN CHAIN TEST")
    print("=" * 70)

    for number, question in enumerate(questions, start=1):
        answer = chain.invoke({"question": question})

        print(f"\nInput {number}: {question}")
        print(f"Output: {answer}")

        results.append(
            f"Input {number}: {question}\n"
            f"Output: {answer}\n"
        )

    return results


def test_memory():
    """Verify conversation history across five turns."""
    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="question",
        return_messages=False,
    )

    prompt = PromptTemplate(
        input_variables=["history", "question"],
        template=(
            "You are an AI/ML mentor.\n"
            "Use the conversation history when it is relevant.\n\n"
            "Conversation history:\n"
            "{history}\n\n"
            "Current question: {question}\n"
            "Answer:"
        ),
    )

    llm = OllamaLLM(model=MODEL_NAME)
    parser = StrOutputParser()

    chain = prompt | llm | parser

    questions = [
        "My name is Ramya.",
        "What is machine learning?",
        "What is supervised learning?",
        "Can you remind me what my name is?",
        "What topic did we discuss about learning?",
    ]

    results = []

    print("\n" + "=" * 70)
    print("TASK 2: CONVERSATION BUFFER MEMORY")
    print("=" * 70)

    for number, question in enumerate(questions, start=1):
        history = memory.load_memory_variables({})["history"]

        answer = chain.invoke(
            {
                "history": history,
                "question": question,
            }
        )

        memory.save_context(
            {"question": question},
            {"answer": answer},
        )

        updated_history = memory.load_memory_variables({})["history"]

        print(f"\nTurn {number}")
        print(f"User: {question}")
        print(f"Assistant: {answer}")
        print("Memory retained:", bool(updated_history))

        results.append(
            f"Turn {number}\n"
            f"User: {question}\n"
            f"Assistant: {answer}\n"
            f"Memory after turn: {updated_history}\n"
        )

    print("\n" + "-" * 70)
    print("FINAL CONVERSATION HISTORY")
    print("-" * 70)
    print(memory.load_memory_variables({})["history"])

    return results


def save_output(chain_results, memory_results):
    """Save execution evidence to the outputs folder."""
    output_file = "outputs/chain_memory_output.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("W6D2 - LANGCHAIN MEMORY & CONVERSATION HISTORY\n")
        file.write("=" * 70 + "\n\n")

        file.write("MODEL\n")
        file.write(f"{MODEL_NAME}\n\n")

        file.write("=" * 70 + "\n")
        file.write("TASK 1: LANGCHAIN CHAIN\n")
        file.write("=" * 70 + "\n\n")

        for result in chain_results:
            file.write(result + "\n")

        file.write("\n")
        file.write("=" * 70 + "\n")
        file.write("TASK 2: CONVERSATION BUFFER MEMORY\n")
        file.write("=" * 70 + "\n\n")

        for result in memory_results:
            file.write(result + "\n")


if __name__ == "__main__":
    chain_results = test_chain()
    memory_results = test_memory()
    save_output(chain_results, memory_results)

    print("\n" + "=" * 70)
    print("Output saved to: outputs/chain_memory_output.txt")
    print("=" * 70)
    