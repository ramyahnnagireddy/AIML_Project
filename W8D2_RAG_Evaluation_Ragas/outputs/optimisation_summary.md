# W8D2 Ragas Optimisation Summary

## Experiment configuration

- Baseline retrieval top-k: 2
- Optimised retrieval top-k: 3
- Baseline chunk size: 700
- Optimised chunk size: 700
- Chunk overlap: 100
- Evaluation samples: 10 fixed questions and reference answers

## Metric comparison

| Metric | Before | After | Change | Valid scores before/after |
|---|---:|---:|---:|---:|
| faithfulness | 0.7125 | 0.7667 | +0.0542 | 10/10, 10/10 |
| answer_relevancy | 0.9153 | 0.9334 | +0.0181 | 10/10, 10/10 |
| context_precision | 0.9000 | 0.9000 | +0.0000 | 10/10, 10/10 |
| context_recall | 0.6825 | 1.0000 | +0.3175 | 10/10, 10/10 |

## Interpretation

Context recall improved by 0.3175 (0.6825 to 1.0000) after changing retrieval top-k.

Scores are averages from the saved Ragas evaluation outputs. A positive change means a higher score; a negative change means a lower score.
