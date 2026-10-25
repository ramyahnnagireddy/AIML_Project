# Production RAG Monitoring Strategy

## 1. Objective

Monitor the Production RAG API for availability, latency, retrieval quality, answer grounding, and operational failures. Use measurable indicators and periodic evaluation to identify regressions before they affect users.

## 2. Current Monitoring

The W8D1 FastAPI service exposes:

- `GET /health` — reports service health information and the configured Ollama URL.
- `GET /metrics` — exposes request counts, error counts, and latency information.
- `POST /query` — accepts a question and returns a RAG answer, retrieved sources, and latency.

The current API metrics are in-process counters. They reset when the process restarts and are not a replacement for persistent production monitoring.

## 3. Key Metrics and Alerts

| Area         | Metric                              | Suggested alert threshold                         | Response                                                   |
| ------------ | ----------------------------------- | ------------------------------------------------- | ---------------------------------------------------------- |
| Availability | Health endpoint failures            | 3 consecutive failures                            | Check API logs and restart or recover the service          |
| Reliability  | HTTP 5xx error rate                 | Above 5% for 5 minutes, with sufficient traffic   | Inspect exceptions and upstream Ollama availability        |
| Latency      | Request p95 latency                 | Above 10 seconds for 10 minutes                   | Check model latency, retrieval time, and resource usage    |
| Retrieval    | Queries returning no useful sources | Above 15% over a representative evaluation window | Check document ingestion, indexing, and retrieval settings |
| Quality      | RAG faithfulness score              | Below 0.75 or a significant drop from baseline    | Review retrieved context, prompts, and model responses     |
| Resources    | CPU and memory usage                | Sustained usage above 85%                         | Inspect container resource limits and workload             |

These are proposed starting thresholds, not measured production baselines. Tune them using real traffic and a representative evaluation dataset.

## 4. Retrieval and Answer Quality

Maintain a versioned evaluation dataset containing questions, expected answers, and relevant source documents.

For each release, measure:

- Retrieval hit rate and Recall@K against known relevant documents.
- Whether answers are supported by retrieved context.
- Answer relevance and correctness against reviewed reference answers.
- Empty or irrelevant retrieval results.
- Latency and error rate for the same evaluation questions.

The current W7D5 evaluation uses deterministic lexical indicators. Treat those results as a baseline only; they are not equivalent to a full Ragas evaluation. Add Ragas metrics when the evaluator and its dependencies are configured and validated.

## 5. Logging and Observability

Record structured events containing:

- Timestamp, request ID, endpoint, HTTP status, and latency.
- Retrieved source identifiers and retrieval scores, when available.
- Model name, prompt/version identifier, and evaluation run identifier.
- Error category and sanitized diagnostic details.

Do not log API keys, credentials, or unnecessary personal information. Apply retention limits and restrict access to logs.

For a larger deployment, export metrics to a persistent monitoring platform and configure dashboards and alerts. Keep application logs separate from evaluation artifacts.

## 6. Incident Response

1. Confirm the alert and determine its scope.
2. Check API logs, `/health`, `/metrics`, and Ollama availability.
3. Reproduce the problem using a small set of known evaluation questions.
4. Identify whether the issue comes from the API, retrieval index, documents, model, or prompt.
5. Roll back the affected application or configuration if necessary.
6. Record the root cause, corrective action, and follow-up evaluation results.

## 7. Re-indexing and Release Triggers

Rebuild the retrieval index when the source-document collection changes. Re-evaluate after changing the model, prompt, chunking strategy, retrieval configuration, or dependencies.

Investigate quality regressions before promoting a release. This RAG workflow does not automatically require model retraining; index rebuilding and configuration updates are the first steps when retrieval quality declines.

## 8. Review Cadence

- Every code change: run linting, automated tests, and the Docker build.
- Every release: run the fixed RAG evaluation dataset and compare results with the baseline.
- Weekly: review errors, latency percentiles, retrieval failures, and resource trends.
- After an incident: document the cause and verify the corrective action with tests.

## 9. Current Limitations

- In-process API counters are not persistent across restarts.
- Health reporting does not by itself guarantee that Ollama can successfully generate a response.
- Suggested thresholds require validation against real measurements.
- Production dashboards, alert delivery, and automated rollback must be configured separately.
