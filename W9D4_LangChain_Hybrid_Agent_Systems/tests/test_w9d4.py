
"""Automated tests for W9D4 LangChain components."""

from app.main import (
    AGENT_TASKS,
    CHAIN_INPUTS,
    MEMORY_TURNS,
    build_chain,
    run_agent_task,
    run_memory_demo,
)
from app.tools import calculator, web_search_stub


def test_chain_has_five_inputs_and_expected_components():
    assert len(CHAIN_INPUTS) == 5
    chain = build_chain()
    assert chain is not None


def test_memory_retains_all_five_conversation_turns():
    results, final_history = run_memory_demo()

    assert len(results) == 5
    for user_message, assistant_message in MEMORY_TURNS:
        assert user_message in final_history
        assert assistant_message in final_history


def test_agent_has_three_demo_tasks():
    assert len(AGENT_TASKS) == 3


def test_agent_routes_search_task_to_search_tool():
    tool_name, result = run_agent_task("Search for an explanation of RAG.")

    assert tool_name == "web_search_stub"
    assert "Retrieval-Augmented Generation" in result


def test_agent_routes_calculation_to_calculator():
    tool_name, result = run_agent_task("Calculate 18 * 24.")

    assert tool_name == "calculator"
    assert "432" in result


def test_calculator_supports_basic_arithmetic():
    assert calculator.invoke({"expression": "10 + 5 * 2"}) == "10 + 5 * 2 = 20"


def test_calculator_rejects_unsupported_expressions():
    result = calculator.invoke({"expression": "__import__('os').getcwd()"})

    assert result.startswith("Calculation error:")


def test_search_stub_is_not_live_web_search():
    result = web_search_stub.invoke({"query": "RAG"})

    assert "Retrieval-Augmented Generation" in result
    assert "offline" in result.lower()
