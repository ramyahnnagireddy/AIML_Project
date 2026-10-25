
"""W9D4: LangChain chain, conversation memory, and a tool-using agent."""

import re
from pathlib import Path

from langchain_classic.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from app.tools import AGENT_TOOLS, calculator, web_search_stub

MODEL_NAME = "qwen2.5:3b"
OUTPUT_PATH = Path("outputs") / "w9d4_evidence.md"

CHAIN_INPUTS = [
    "Explain RAG in one sentence.",
    "Explain why MLflow is useful in machine learning.",
    "Give one benefit of AI in healthcare.",
    "Explain one basic cybersecurity practice.",
    "Explain what LangChain helps developers build.",
]

MEMORY_TURNS = [
    ("My project uses RAG.", "RAG retrieves relevant context for a language model."),
    ("Which tool tracks experiments?", "MLflow can track machine-learning experiments."),
    ("What model am I using today?", "You are using the local qwen2.5:3b model."),
    ("Why keep conversation history?", "History helps preserve context across turns."),
    ("Summarize my project focus.", "Your focus is a LangChain chain, memory, and tools."),
]

AGENT_TASKS = [
    "Search for an explanation of RAG.",
    "Calculate 18 * 24.",
    "Search for the purpose of MLflow.",
]


def build_chain():
    """Build the PromptTemplate -> Ollama -> StrOutputParser chain."""
    prompt = PromptTemplate.from_template(
        "You are a concise technical learning assistant. "
        "Answer the following question in 1-3 sentences using clear language.\n"
        "Question: {question}\n"
        "Answer:"
    )
    llm = OllamaLLM(model=MODEL_NAME, temperature=0)
    return prompt | llm | StrOutputParser()


def run_chain_demo() -> list[tuple[str, str]]:
    """Invoke the chain with five different inputs."""
    chain = build_chain()
    results = []

    print("\n=== CHAIN DEMO: FIVE INPUTS ===")
    for index, question in enumerate(CHAIN_INPUTS, start=1):
        answer = chain.invoke({"question": question})
        results.append((question, answer.strip()))
        print(f"\nInput {index}: {question}")
        print(f"Output {index}: {answer.strip()}")

    return results


def run_memory_demo() -> tuple[list[dict[str, str]], str]:
    """Save five turns and verify earlier turns remain in the buffer."""
    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="input",
        output_key="output",
        return_messages=False,
    )
    results = []

    print("\n=== CONVERSATION MEMORY: FIVE TURNS ===")
    for turn_number, (user_message, assistant_message) in enumerate(
        MEMORY_TURNS, start=1
    ):
        memory.save_context(
            {"input": user_message},
            {"output": assistant_message},
        )
        history = memory.load_memory_variables({})["history"]
        results.append(
            {
                "turn": str(turn_number),
                "user": user_message,
                "assistant": assistant_message,
                "history_length": str(len(history)),
            }
        )
        print(f"Turn {turn_number}: {user_message}")
        print(f"Assistant: {assistant_message}")
        print(f"History characters: {len(history)}")

    final_history = memory.load_memory_variables({})["history"]
    all_turns_retained = all(
        user_message in final_history
        and assistant_message in final_history
        for user_message, assistant_message in MEMORY_TURNS
    )
    if not all_turns_retained:
        raise AssertionError("Conversation memory did not retain all five turns.")

    print(f"All five turns retained: {all_turns_retained}")
    return results, final_history


def run_agent_task(task: str) -> tuple[str, str]:
    """Route a task to one of the two registered LangChain tools."""
    calculator_match = re.fullmatch(
        r"\s*calculate\s+(.+?)\s*[?.!]?\s*",
        task,
        flags=re.IGNORECASE,
    )

    if calculator_match:
        tool_result = calculator.invoke(
            {"expression": calculator_match.group(1).rstrip("?.! ")}
        )
        return "calculator", str(tool_result)

    search_query = re.sub(
        r"^\s*search\s+for\s+",
        "",
        task,
        flags=re.IGNORECASE,
    ).strip(" ?.!")
    tool_result = web_search_stub.invoke({"query": search_query})
    return "web_search_stub", str(tool_result)


def run_agent_demo() -> list[dict[str, str]]:
    """Run three tasks and record which tool handled each one."""
    results = []
    registered_tool_names = {tool.name for tool in AGENT_TOOLS}

    print("\n=== SIMPLE TOOL-USING AGENT: THREE TASKS ===")
    for index, task in enumerate(AGENT_TASKS, start=1):
        tool_name, result = run_agent_task(task)
        if tool_name not in registered_tool_names:
            raise AssertionError(f"Unregistered tool selected: {tool_name}")

        results.append(
            {"task": task, "tool": tool_name, "result": result}
        )
        print(f"\nTask {index}: {task}")
        print(f"Selected tool: {tool_name}")
        print(f"Result: {result}")

    return results


def write_evidence(
    chain_results: list[tuple[str, str]],
    memory_results: list[dict[str, str]],
    agent_results: list[dict[str, str]],
    final_history: str,
) -> Path:
    """Write reproducible demonstration evidence to Markdown."""
    output_path = Path(__file__).resolve().parent.parent / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# W9D4 — LangChain Hybrid Agent Systems",
        "",
        f"- Model: `{MODEL_NAME}`",
        "- Chain: `PromptTemplate → OllamaLLM → StrOutputParser`",
        "- Memory: `ConversationBufferMemory`",
        "- Tools: `web_search_stub`, `calculator`",
        "",
        "## 1. Chain — five inputs",
        "",
    ]

    for index, (question, answer) in enumerate(chain_results, start=1):
        lines.extend(
            [
                f"### Input {index}",
                f"**Question:** {question}",
                "",
                f"**Output:** {answer}",
                "",
            ]
        )

    lines.extend(["## 2. Conversation memory — five turns", ""])
    for turn in memory_results:
        lines.extend(
            [
                f"### Turn {turn['turn']}",
                f"- User: {turn['user']}",
                f"- Assistant: {turn['assistant']}",
                f"- History characters after turn: {turn['history_length']}",
                "",
            ]
        )

    lines.extend(
        [
            ("**Verification:** All five user messages and assistant responses "
            "were found in the final conversation buffer."),
            "",
            "## 3. Tool-using agent — three tasks",
            "",
        ]
    )

    for index, task in enumerate(agent_results, start=1):
        lines.extend(
            [
                f"### Task {index}",
                f"- Task: {task['task']}",
                f"- Selected tool: `{task['tool']}`",
                f"- Result: {task['result']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Final memory buffer",
            "",
            "```text",
            final_history,
            "```",
            "",
            "## Notes",
            "",
            "- The search tool is a predefined offline stub, not live web search.",
            "- The agent uses an explicit routing policy to select tools.",
            "- LLM responses are generated locally using Ollama.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    chain_results = run_chain_demo()
    memory_results, final_history = run_memory_demo()
    agent_results = run_agent_demo()

    evidence_path = write_evidence(
        chain_results,
        memory_results,
        agent_results,
        final_history,
    )
    print(f"\nEvidence saved to: {evidence_path}")
    print("\nW9D4 demonstration completed successfully.")


if __name__ == "__main__":
    main()
