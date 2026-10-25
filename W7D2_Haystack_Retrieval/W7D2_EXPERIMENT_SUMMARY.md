# W7D2 Haystack Retrieval Experiment

## Objective

Compare BM25 lexical retrieval with dense semantic retrieval using Haystack on the same set of 5 PDF documents and 10 questions.

## Documents

- computer_vision.pdf
- deep_learning.pdf
- machine_learning.pdf
- mlops.pdf
- natural_language_processing.pdf

## Retrieval Methods

### BM25

Haystack InMemoryBM25Retriever was used to retrieve the top 3 documents for each question.

### Dense Retrieval

Haystack InMemoryEmbeddingRetriever was used with the sentence-transformers/all-MiniLM-L6-v2 embedding model.

## Evaluation

| Method | Top-1 Precision | Top-3 Hit Rate |
|---|---:|---:|
| BM25 | 90% | 100% |
| Dense Retrieval | 100% | 100% |

## Observation

BM25 retrieved the expected document at rank 1 for 9 out of 10 questions. The only top-1 miss was the classification-metrics question, where the expected machine-learning document appeared at rank 2.

Dense retrieval returned the expected document at rank 1 for all 10 questions.

Both methods achieved a 100% top-3 hit rate on this controlled 10-question evaluation set.

## Evidence

- outputs/bm25_results.json
- outputs/dense_results.json
- outputs/retrieval_comparison.json
- outputs/manual_evaluation.md

## Conclusion

The experiment demonstrates both lexical BM25 retrieval and semantic dense retrieval in Haystack and provides a direct comparison using the same documents and questions.
