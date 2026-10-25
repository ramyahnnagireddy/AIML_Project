
"""Web search and restricted arithmetic tools for W9D2."""

import ast
import operator
import re
from html import unescape

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    """Input schema for Wikipedia search."""

    query: str = Field(
        ...,
        min_length=1,
        description="A topic or question to search on Wikipedia.",
    )


class CalculatorInput(BaseModel):
    """Input schema for safe arithmetic evaluation."""

    expression: str = Field(
        ...,
        min_length=1,
        description="An arithmetic expression such as (25 + 15) / 4.",
    )


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calculate(expression: str) -> float | int:
    """Evaluate basic arithmetic without eval or arbitrary code execution."""
    if not expression.strip() or len(expression) > 100:
        raise ValueError("Expression must contain 1–100 characters.")

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                if abs(node.value) > 1_000_000_000:
                    raise ValueError("Number exceeds the allowed limit.")
                return node.value
            raise ValueError("Only numeric constants are allowed.")

        if isinstance(node, ast.BinOp):
            operation = _BINARY_OPERATORS.get(type(node.op))
            if operation is None:
                raise ValueError("Operator is not allowed.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent magnitude must not exceed 10.")

            if (
                isinstance(node.op, (ast.Mult, ast.Pow))
                and (abs(left) > 1_000_000 or abs(right) > 1_000_000)
            ):
                raise ValueError("Operand exceeds the allowed limit.")

            result = operation(left, right)

            if abs(result) > 1e12:
                raise ValueError("Result exceeds the allowed limit.")

            return result

        if isinstance(node, ast.UnaryOp):
            operation = _UNARY_OPERATORS.get(type(node.op))
            if operation is None:
                raise ValueError("Unary operator is not allowed.")
            return operation(evaluate(node.operand))

        raise ValueError("Only basic arithmetic is supported.")

    return evaluate(tree)


class CalculatorTool(BaseTool):
    """A calculator that evaluates restricted arithmetic expressions."""

    name: str = "Safe Arithmetic Calculator"
    description: str = (
        "Calculate basic arithmetic expressions using numbers, "
        "parentheses, addition, subtraction, multiplication, division, "
        "floor division, modulo, and limited exponentiation."
    )
    args_schema: type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        try:
            return str(safe_calculate(expression))
        except (ValueError, SyntaxError, TypeError, ArithmeticError) as exc:
            return f"Calculation rejected: {exc}"


class WikipediaSearchTool(BaseTool):
    """Search Wikipedia and return actual article titles and URLs."""

    name: str = "Wikipedia Web Search"
    description: str = (
        "Search Wikipedia for a topic. Returns actual article titles, "
        "search snippets, and URLs supplied by the Wikipedia API."
    )
    args_schema: type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "Search rejected: query cannot be empty."

        try:
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": 5,
                },
                headers={
                    "User-Agent": "CynarisW9D2ResearchCrew/1.0"
                },
                timeout=15,
            )
            response.raise_for_status()
            results = response.json().get("query", {}).get("search", [])
        except (requests.RequestException, ValueError) as exc:
            return f"Web search unavailable: {exc}"

        if not results:
            return f"No Wikipedia results found for: {query}"

        formatted = []

        for item in results:
            page_id = item.get("pageid")
            if not page_id:
                continue

            title = item.get("title", "Untitled")
            snippet = unescape(
                re.sub(r"<[^>]*>", "", item.get("snippet", ""))
            ).strip()
            url = f"https://en.wikipedia.org/?curid={page_id}"

            formatted.append(
                f"Title: {title}\n"
                f"Snippet: {snippet}\n"
                f"URL: {url}"
            )

        if not formatted:
            return "Wikipedia returned no usable article results."

        return "\n\n".join(formatted)
    