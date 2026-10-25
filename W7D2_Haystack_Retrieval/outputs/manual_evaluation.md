# W7D2 Manual Retrieval Evaluation

## Evaluation Method

The same 10 questions were evaluated against five indexed PDF documents.
Each question was assigned an expected relevant source document.
Top-1 precision measures whether the expected document was ranked first.
Top-3 hit rate measures whether the expected document appeared within the top three results.

## Results

| Q | Expected Source | BM25 Rank | BM25 Top-1 | Dense Rank | Dense Top-1 |
|---|---|---:|---|---:|---|
| 1 | machine_learning.pdf | 1 | PASS | 1 | PASS |
| 2 | machine_learning.pdf | 2 | MISS | 1 | PASS |
| 3 | deep_learning.pdf | 1 | PASS | 1 | PASS |
| 4 | deep_learning.pdf | 1 | PASS | 1 | PASS |
| 5 | natural_language_processing.pdf | 1 | PASS | 1 | PASS |
| 6 | natural_language_processing.pdf | 1 | PASS | 1 | PASS |
| 7 | computer_vision.pdf | 1 | PASS | 1 | PASS |
| 8 | computer_vision.pdf | 1 | PASS | 1 | PASS |
| 9 | mlops.pdf | 1 | PASS | 1 | PASS |
| 10 | mlops.pdf | 1 | PASS | 1 | PASS |

## Summary

- BM25 Top-1 Precision: 90.0% (9/10)
- BM25 Top-3 Hit Rate: 100.0% (10/10)
- Dense Top-1 Precision: 100.0% (10/10)
- Dense Top-3 Hit Rate: 100.0% (10/10)

## Manual Observation

BM25 retrieved the expected document within the top three results for all 10 questions.
For Question 2, the expected machine_learning.pdf was ranked second by BM25.
Dense retrieval ranked the expected document first for all 10 questions.
These observations apply to this controlled five-document, ten-question evaluation set.