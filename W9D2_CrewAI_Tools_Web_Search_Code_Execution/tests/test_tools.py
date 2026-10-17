
"""Tests for W9D2 tools."""

import pytest
from tools import safe_calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3", 5),
        ("(25 + 15) / 4", 10),
        ("3 * (4 + 2)", 18),
        ("2 ** 3", 8),
        ("10 % 3", 1),
        ("-5 + 8", 3),
    ],
)
def test_safe_calculate_valid_expressions(expression, expected):
    assert safe_calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "import os",
        "__import__('os').system('whoami')",
        "open('secret.txt').read()",
        "2 ** 100",
        "1 / 0",
        "[1, 2, 3]",
    ],
)
def test_safe_calculate_rejects_unsafe_expressions(expression):
    with pytest.raises((ValueError, SyntaxError, TypeError, ArithmeticError)):
        safe_calculate(expression)


def test_safe_calculate_rejects_oversized_expression():
    with pytest.raises(ValueError):
        safe_calculate("1" * 101)
        