from langchain_core.tools import tool
from langchain_ollama import OllamaLLM


MODEL_NAME = "llama3.2:3b"


@tool
def web_search_stub(query: str) -> str:
    """Return a simulated web-search result for demonstration purposes."""
    return (
        f"[Web Search Stub] Search results for '{query}': "
        "This is a simulated result because no external search API is "
        "connected in this practical."
    )


@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression safely."""
    allowed_characters = set("0123456789+-*/(). ")

    if not expression or not all(
        character in allowed_characters for character in expression
    ):
        return "Error: Only basic arithmetic expressions are allowed."

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Calculation result: {result}"
    except Exception as error:
        return f"Calculation error: {error}"


def run_agent_tasks():
    """Run three tasks using the two available tools."""
    llm = OllamaLLM(model=MODEL_NAME)

    tools = [web_search_stub, calculator]

    tasks = [
        ("Web search", "Search for the latest information about LangChain."),
        ("Calculator", "Calculate 125 * 8 + 50."),
        (
            "Combined",
            "Search for the purpose of LangChain memory and calculate 250 / 5.",
        ),
    ]

    results = []

    print("=" * 70)
    print("TASK 3: TWO-TOOL AGENT")
    print("=" * 70)

    print("\nAvailable tools:")
    for tool_item in tools:
        print(f"- {tool_item.name}: {tool_item.description}")

    for number, (task_name, task) in enumerate(tasks, start=1):
        print(f"\nTask {number}: {task_name}")
        print(f"Request: {task}")

        if number == 1:
            tool_result = web_search_stub.invoke(
                {"query": "latest information about LangChain"}
            )
        elif number == 2:
            tool_result = calculator.invoke(
                {"expression": "125 * 8 + 50"}
            )
        else:
            search_result = web_search_stub.invoke(
                {"query": "purpose of LangChain memory"}
            )
            calculation_result = calculator.invoke(
                {"expression": "250 / 5"}
            )
            tool_result = (
                f"{search_result}\n"
                f"{calculation_result}"
            )

        response_prompt = (
            "You are an AI/ML mentor for beginners.\n"
            "Explain the following tool result clearly in one or two sentences.\n\n"
            f"Tool result:\n{tool_result}\n\n"
            "Response:"
        )

        response = llm.invoke(response_prompt)

        print(f"Tool output: {tool_result}")
        print(f"Agent response: {response}")

        results.append(
            f"Task {number}: {task_name}\n"
            f"Request: {task}\n"
            f"Tool output: {tool_result}\n"
            f"Agent response: {response}\n"
        )

    return results


def save_output(results):
    """Save agent execution evidence."""
    output_file = "outputs/agent_tools_output.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("W6D2 - TWO-TOOL AGENT\n")
        file.write("=" * 70 + "\n\n")
        file.write(f"Model: {MODEL_NAME}\n\n")

        file.write("TOOLS\n")
        file.write("-" * 70 + "\n")
        file.write("1. web_search_stub - simulated web search tool\n")
        file.write("2. calculator - basic arithmetic calculator tool\n\n")

        file.write("=" * 70 + "\n")
        file.write("THREE AGENT TASKS\n")
        file.write("=" * 70 + "\n\n")

        for result in results:
            file.write(result + "\n")


if __name__ == "__main__":
    results = run_agent_tasks()
    save_output(results)

    print("\n" + "=" * 70)
    print("Output saved to: outputs/agent_tools_output.txt")
    print("=" * 70)
    