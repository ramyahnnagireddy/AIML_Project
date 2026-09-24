from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


MODEL_NAME = "llama3.2:3b"


def build_chain():
    """Build PromptTemplate -> Ollama LLM -> StrOutputParser chain."""
    prompt = PromptTemplate.from_template(
        "You are a helpful AI/ML assistant.\n"
        "Answer the following question clearly and briefly.\n\n"
        "Question: {question}"
    )

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    parser = StrOutputParser()

    return prompt | llm | parser


def run_chain_tests():
    """Test the chain with five different inputs."""
    chain = build_chain()

    questions = [
        "What is machine learning?",
        "What is supervised learning?",
        "What is a neural network?",
        "What is overfitting?",
        "What is a vector database?",
    ]

    results = []

    for index, question in enumerate(questions, start=1):
        answer = chain.invoke({"question": question})

        results.append(
            f"Input {index}: {question}\n"
            f"Output: {answer}\n"
        )

    return results


def run_conversation_memory_demo():
    """
    Demonstrate conversation history across five turns.

    The history is maintained explicitly and supplied to the prompt
    so that the chatbot can use previous turns when answering.
    """
    prompt = PromptTemplate.from_template(
        "You are a helpful AI/ML chatbot.\n"
        "Use the conversation history to answer the current question.\n\n"
        "Conversation history:\n"
        "{history}\n\n"
        "Current question: {question}\n"
        "Answer:"
    )

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser

    conversation = []

    turns = [
        "My name is Ramya.",
        "I am studying computer science engineering.",
        "What field am I studying?",
        "What should I learn first for an AI career?",
        "Can you summarize what you know about me from this conversation?",
    ]

    results = []

    for turn_number, question in enumerate(turns, start=1):
        history = "\n".join(
            f"User: {user_message}\nAssistant: {assistant_message}"
            for user_message, assistant_message in conversation
        )

        answer = chain.invoke(
            {
                "history": history if history else "No previous conversation.",
                "question": question,
            }
        )

        conversation.append((question, answer))

        results.append(
            f"Turn {turn_number}\n"
            f"User: {question}\n"
            f"Assistant: {answer}\n"
        )

    return results


def main():
    print("=" * 70)
    print("W6D5 - LANGCHAIN CHAIN TEST")
    print("=" * 70)

    chain_results = run_chain_tests()

    for result in chain_results:
        print(result)

    with open(
        "W6D5_Document_Chatbot_LangChain/outputs/chain_output.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("W6D5 - LANGCHAIN CHAIN TEST\n\n")
        for result in chain_results:
            file.write(result + "\n")

    print("=" * 70)
    print("W6D5 - CONVERSATION MEMORY TEST")
    print("=" * 70)

    memory_results = run_conversation_memory_demo()

    for result in memory_results:
        print(result)

    with open(
        "W6D5_Document_Chatbot_LangChain/outputs/memory_output.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("W6D5 - CONVERSATION MEMORY TEST\n\n")
        for result in memory_results:
            file.write(result + "\n")

    print("=" * 70)
    print("CHAIN AND MEMORY TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
    