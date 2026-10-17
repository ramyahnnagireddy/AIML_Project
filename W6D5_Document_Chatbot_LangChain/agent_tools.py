from langchain_core.tools import tool
from langchain_ollama import ChatOllama


MODEL_NAME = "llama3.2:3b"


@tool
def web_search_stub(query: str) -> str:
    """Return a simulated web search result for a query."""
    return (
        f"Web search stub result for '{query}': "
        "This is a simulated search result used for the W6D5 "
        "LangChain agent demonstration."
    )


@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression."""
    allowed_characters = "0123456789+-*/(). "

    if not expression or any(
        character not in allowed_characters
        for character in expression
    ):
        return "Error: only basic arithmetic expressions are supported."

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as exc:
        return f"Error calculating expression: {exc}"


def run_tool_tests():
    """Directly test both tools before running the agent."""
    print("=" * 70)
    print("W6D5 - TOOL TESTS")
    print("=" * 70)

    print("\nWeb Search Stub:")
    print(web_search_stub.invoke({"query": "LangChain agents"}))

    print("\nCalculator:")
    print(calculator.invoke({"expression": "125 * 8 + 50"}))


def build_agent():
    """
    Build a simple LangChain agent with two tools.
    """
    from langchain.agents import create_agent

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0,
    )

    tools = [
        web_search_stub,
        calculator,
    ]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful AI/ML assistant. "
            "Use the web_search_stub tool when a web search is requested. "
            "Use the calculator tool for arithmetic calculations. "
            "Answer clearly and briefly."
        ),
    )


def run_agent_task(agent, task: str) -> str:
    """Run one task through the agent."""
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": task,
                }
            ]
        }
    )

    messages = response.get("messages", [])

    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            content = getattr(message, "content", "")
            if content:
                return content

    return "No final agent response was returned."


def run_agent_tests():
    """Run three tasks using the two-tool agent."""
    agent = build_agent()

    tasks = [
        "Use the web search tool to search for LangChain agents.",
        "Calculate 125 * 24.",
        "Use the calculator to calculate 48 / 6.",
    ]

    results = []

    for index, task in enumerate(tasks, start=1):
        print(f"\nTask {index}: {task}")

        answer = run_agent_task(agent, task)

        print(f"Agent: {answer}")

        results.append(
            f"Task {index}: {task}\n"
            f"Agent: {answer}\n"
        )

    return results


def main():
    run_tool_tests()

    print("\n" + "=" * 70)
    print("W6D5 - TWO-TOOL AGENT TEST")
    print("=" * 70)

    results = run_agent_tests()

    with open(
        "W6D5_Document_Chatbot_LangChain/outputs/agent_output.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("W6D5 - TWO-TOOL AGENT TEST\n\n")

        for result in results:
            file.write(result + "\n")

    print("=" * 70)
    print("AGENT TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
    