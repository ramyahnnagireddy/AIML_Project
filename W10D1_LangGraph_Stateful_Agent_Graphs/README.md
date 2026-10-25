#W10D1 — LangGraph Stateful Agent Graphs

## Objective

Build and test a three-node LangGraph with conditional routing and a
human-in-the-loop interrupt.

## Graph design

1. `classify` — classifies the input as `general` or `needs_review`.
2. `route` — requests a human decision using a LangGraph interrupt.
3. `respond` — generates the final response.

Conditional edges route general inputs directly to `respond` and
review-required inputs to `route`. The review path pauses until a
human decision is supplied.

## Implementation

- LangGraph `StateGraph`
- Typed state shared between nodes
- Conditional edges
- `InMemorySaver` checkpointing
- `interrupt()` and `Command(resume=...)`
- Five demonstration inputs
- Pytest automated tests
- JSON execution evidence

## Run

From this project directory with the repository virtual environment
activated:

```powershell
python main.py
python -m pytest tests -v
python -m ruff check .
```

## Evidence

The demonstration writes `outputs/run_evidence.json`.

The graph uses deterministic classification and response logic.
It does not call an external LLM or perform real-world actions.
Human approval in this demonstration is a workflow checkpoint,
not a substitute for production authorization controls.
