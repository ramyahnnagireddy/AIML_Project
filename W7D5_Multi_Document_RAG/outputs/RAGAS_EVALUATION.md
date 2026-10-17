# W7D5 RAG Evaluation

- Evaluation time: 2026-10-14T18:18:45.262159+00:00
- Samples evaluated: 5
- Method: deterministic lexical indicators

## Aggregate Results

| Indicator | Average |
|---|---:|
| Context support indicator | 0.5349 |
| Question-term overlap indicator | 0.5967 |

## Per-sample Results

| Sample | Context support | Question-term overlap |
|---:|---:|---:|
| 1 | 0.8182 | 0.75 |
| 2 | 0.4783 | 0.4 |
| 3 | 0.3529 | 0.8333 |
| 4 | 0.5676 | 0.4 |
| 5 | 0.4576 | 0.6 |

## Limitations

- These are lexical indicators, not Ragas Faithfulness or AnswerRelevancy scores.
- Token overlap does not establish semantic correctness or factual entailment.
- Results should be interpreted as preliminary diagnostics only.
