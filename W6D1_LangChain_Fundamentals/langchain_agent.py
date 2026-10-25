"""
W6D1 - LangChain Fundamentals
Task 3: Simple LangChain Agent

Agent tools:
1. Web search stub
2. Calculator

Runs 3 test tasks.
"""

import ast
import operator as op

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


MODEL_NAME = "llama3.2:3b"


# Allowed arithmetic operations for the calculator.
_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
}


def _safe_calculate(node):
    """Evaluate a mathematical expression safely."""

    if isinstance(node, ast.Constant) and isinstance(
        node.value, (int, float)
    ):
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        operand = _safe_calculate(node.operand)
        return _ALLOWED_OPERATORS[type(node.op)](operand)

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_calculate(node.left)
        right = _safe_calculate(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)

    raise ValueError("Unsupported mathematical expression.")


@tool
def web_search_stub(query: str) -> str:
    """Return a simulated web-search result for demonstration purposes."""

    return (
        f"[WEB SEARCH STUB] Simulated search result for: '{query}'. "
        "This is a stub and does not perform a real web search."
    )


@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression safely."""

    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_calculate(tree.body)
        return str(result)

    except Exception as error:
        return f"Calculator error: {error}"


def build_agent():
    """Create and return the LangChain agent."""

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    tools = [
        web_search_stub,
        calculator,
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful AI/ML mentor. "
            "Use the available tools when they are useful. "
            "For arithmetic calculations, use the calculator tool. "
            "For web-search requests, use the web_search_stub tool. "
            "Keep final answers clear and concise."
        ),
    )

    return agent


def main():
    """Run the agent with 3 test tasks."""

    agent = build_agent()

    tasks = [
        "Calculate 125 * 8 + 50.",
        "Use the web search tool to find information about LangChain.",
        "Calculate (250 / 5) + 30."
    ]

    print("=" * 70)
    print("W6D1 - LangChain Agent Test")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print("Tools: web_search_stub, calculator")
    print("Tasks: 3")
    print("=" * 70)

    for task_number, task in enumerate(tasks, start=1):
        print(f"\nTASK {task_number}")
        print(f"INPUT: {task}")
        print("-" * 70)

        try:
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": task,
                        }
                    ]
                }
            )

            messages = result.get("messages", [])

            print("AGENT OUTPUT:")

            if messages:
                final_message = messages[-1]
                print(final_message.content)

            print(f"Messages in execution: {len(messages)}")

        except Exception as error:
            print(f"ERROR: {error}")


if __name__ == "__main__":
    main()
    