from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


def build_chain():
    """Build a PromptTemplate -> Ollama LLM -> OutputParser chain."""

    prompt = PromptTemplate(
        input_variables=["topic"],
        template=(
            "You are an AI/ML mentor for beginners. "
            "Explain the following topic in simple terms with one short example:\n\n"
            "{topic}"
        ),
    )

    llm = OllamaLLM(model="llama3.2:3b")

    parser = StrOutputParser()

    chain = prompt | llm | parser

    return chain


def main():
    chain = build_chain()

    test_inputs = [
        "What is machine learning?",
        "What is a neural network?",
        "What is overfitting?",
        "What is an API?",
        "What is LangChain?",
    ]

    output_lines = [
        "W6D3 TASK 1 - LANGCHAIN CHAIN",
        "=" * 50,
        "Pipeline: PromptTemplate -> Ollama LLM -> StrOutputParser",
        "Model: llama3.2:3b",
        "",
    ]

    for index, topic in enumerate(test_inputs, start=1):
        print(f"\nInput {index}: {topic}")
        response = chain.invoke({"topic": topic})

        print("Response:")
        print(response)

        output_lines.append(f"INPUT {index}: {topic}")
        output_lines.append("RESPONSE:")
        output_lines.append(response)
        output_lines.append("-" * 50)

    with open(
        "W6D3_LangChain_Tools_Agents/outputs/chain_output.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(output_lines))

    print("\nChain test completed.")
    print("Evidence saved to outputs/chain_output.txt")


if __name__ == "__main__":
    main()
    