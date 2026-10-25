
import json

from app.graph import build_graph, run_with_human_review

SAMPLE_INPUTS = [
    "I have a Python error in my code.",
    "I need help with a payment refund.",
    "This is an urgent emergency and needs attention.",
    "Can you explain how AI works?",
    "My invoice has an incorrect amount.",
]


def main() -> None:
    graph = build_graph()

    print("=== W10D2: Conditional Routing Demo ===")

    for index, user_input in enumerate(SAMPLE_INPUTS, start=1):
        config = {
            "configurable": {
                "thread_id": f"w10d2-test-{index}",
            }
        }

        # Disable interrupts for the five-input routing demonstration.
        # The separate demo below tests human approval and resume.
        result = graph.invoke(
            {"user_input": user_input},
            config=config,
            interrupt_before=["respond"],
        )

        print(f"\nTest {index}: {user_input}")
        print(f"Category: {result['category']}")
        print(f"Route: {result.get('route', 'direct_response')}")
        print(f"Response node reached: {'response' in result}")

    print("\n=== Human-in-the-Loop Demo ===")
    final_result = run_with_human_review(
        "Urgent: a critical system error needs attention."
    )
    print("Resumed result:")
    print(json.dumps(final_result, indent=2))


if __name__ == "__main__":
    main()
    