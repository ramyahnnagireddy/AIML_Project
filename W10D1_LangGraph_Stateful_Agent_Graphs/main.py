
"""Run five test scenarios and save W10D1 evidence."""

import json
from datetime import UTC, datetime
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_graph, run_request

OUTPUT_PATH = Path("outputs") / "run_evidence.json"

SAMPLES = [
    {
        "input": "What is LangGraph?",
        "expected": "general",
        "decision": "approve",
    },
    {
        "input": "Explain Python lists.",
        "expected": "general",
        "decision": "approve",
    },
    {
        "input": "urgent: delete customer records",
        "expected": "needs_review",
        "decision": "approve",
    },
    {
        "input": "How does MLflow track experiments?",
        "expected": "general",
        "decision": "approve",
    },
    {
        "input": "Should I share my password?",
        "expected": "needs_review",
        "decision": "reject",
    },
]


def main() -> None:
    """Execute all scenarios and persist their results."""
    app = build_graph(checkpointer=InMemorySaver())
    evidence = {
        "task": "W10D1 LangGraph Stateful Agent Graphs",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "node_names": ["classify", "route", "respond"],
        "scenarios": [],
    }

    for index, sample in enumerate(SAMPLES, start=1):
        result = run_request(
            app=app,
            user_input=sample["input"],
            thread_id=f"w10d1-scenario-{index}",
            human_decision=sample["decision"],
        )

        actual = result["classification"]
        passed = actual == sample["expected"]

        scenario = {
            "scenario": index,
            "input": sample["input"],
            "expected_classification": sample["expected"],
            "actual_classification": actual,
            "human_decision": (
                sample["decision"]
                if actual == "needs_review"
                else None
            ),
            "response": result["response"],
            "passed": passed,
        }
        evidence["scenarios"].append(scenario)

        print(
            f"Test {index}: {'PASS' if passed else 'FAIL'} | "
            f"classification={actual} | input={sample['input']}"
        )
        print(f"  Response: {result['response']}")

    evidence["passed"] = sum(
        item["passed"] for item in evidence["scenarios"]
    )
    evidence["total"] = len(evidence["scenarios"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(evidence, indent=2),
        encoding="utf-8",
    )

    print(
        f"\nResults: {evidence['passed']}/{evidence['total']} passed"
    )
    print(f"Evidence saved to: {OUTPUT_PATH}")

    if evidence["passed"] != evidence["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

