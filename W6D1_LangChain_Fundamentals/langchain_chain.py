"""
W6D1 - LangChain Fundamentals
Task 1: PromptTemplate -> Ollama LLM -> StrOutputParser

Tests the LangChain chain with 5 different inputs.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


MODEL_NAME = "llama3.2:3b"


def build_chain():
    """Build and return the LangChain prompt -> LLM -> parser chain."""

    prompt = PromptTemplate.from_template(
        """
You are an AI/ML mentor for beginners.

Explain the following topic in simple and clear language.
Give one small example when useful.

Topic: {topic}
"""
    )

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain


def main():
    """Run the chain with 5 test inputs."""

    chain = build_chain()

    test_inputs = [
        "What is machine learning?",
        "What is a Python function?",
        "What is a database?",
        "What is an API?",
        "What is overfitting in machine learning?"
    ]

    print("=" * 70)
    print("W6D1 - LangChain Chain Test")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print("Flow: PromptTemplate -> ChatOllama -> StrOutputParser")
    print("=" * 70)

    for index, topic in enumerate(test_inputs, start=1):
        print(f"\nTEST {index}")
        print(f"INPUT: {topic}")
        print("-" * 70)

        try:
            response = chain.invoke({"topic": topic})
            print("OUTPUT:")
            print(response)

        except Exception as error:
            print(f"ERROR: {error}")


if __name__ == "__main__":
    main()
    