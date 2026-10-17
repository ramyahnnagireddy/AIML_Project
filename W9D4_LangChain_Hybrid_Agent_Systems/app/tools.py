
"""Tools for the W9D4 LangChain hybrid agent demonstration."""

import ast
import operator
from collections.abc import Callable

from langchain_core.tools import tool

_SEARCH_KNOWLEDGE = {
    "rag": (
        "Retrieval-Augmented Generation (RAG) retrieves relevant documents "
        "and provides their context to a language model before it answers."
    ),
    "mlflow": (
        "MLflow supports experiment tracking, model evaluation, and "
        "management of machine-learning model lifecycles."
    ),
    "cybersecurity": (
        "Cybersecurity practices include access control, secure software "
        "development, monitoring, and vulnerability management."
    ),
    "healthcare": (
        "AI in healthcare can support clinical decision-making and medical "
        "research; outputs require appropriate human oversight."
    ),
    "langchain": (
        "LangChain provides components for building applications with "
        "language models, including prompts, parsers, and tools."
    ),
}


@tool
def web_search_stub(query: str) -> str:
    """Search a small offline knowledge base; this is not live web search."""
    query_lower = query.lower()

    for keyword, result in _SEARCH_KNOWLEDGE.items():
        if keyword in query_lower:
            return f"Offline search result for '{query}': {result}"

    return (
        f"Offline search result for '{query}': No exact topic match was found. "
        "This demonstration uses a small predefined knowledge base, "
        "not a live internet search."
    )


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_node(node: ast.AST) -> float:
    """Evaluate only numeric literals and approved arithmetic operations."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent magnitude must not exceed 10.")

        result = _BINARY_OPERATORS[type(node.op)](left, right)

        if abs(result) > 1e100:
            raise ValueError("Result is too large.")

        return result

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand))

    raise ValueError("Only basic numeric arithmetic is supported.")


@tool
def calculator(expression: str) -> str:
    """Calculate a numeric expression using basic arithmetic and parentheses."""
    try:
        if len(expression) > 100:
            return "Error: expression is too long."

        tree = ast.parse(expression, mode="eval")
        result = _evaluate_node(tree.body)
        display_result = int(result) if result.is_integer() else result
        return f"{expression} = {display_result}"
    except (
        SyntaxError,
        ValueError,
        TypeError,
        ZeroDivisionError,
        OverflowError,
    ) as exc:
        return f"Calculation error: {exc}"


AGENT_TOOLS = [web_search_stub, calculator]
