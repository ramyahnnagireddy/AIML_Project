# W7D1 Haystack Retrieval Experiment

## Experiment Setup

- Framework: Haystack
- Documents indexed: 5 PDF documents
- Queries evaluated: 10
- Retrieval methods:
  - InMemoryBM25Retriever
  - InMemoryEmbeddingRetriever
- Dense embedding model: sentence-transformers/all-MiniLM-L6-v2

## Evaluation Method

The same 10 questions were submitted to both retrieval methods. For each question, the top retrieved documents and retrieval scores were recorded.

## Observations

BM25 performed lexical matching based on terms appearing in the documents.

Dense retrieval used vector embeddings to identify semantically related documents.

Both methods produced the same top-ranked document for several questions, while other questions produced different top-ranked documents.

The results demonstrate that lexical and semantic retrieval can produce different rankings for the same query.

## Evidence

Detailed retrieval results are available in:

outputs/retrieval_results.json

Manual comparison is available in:

outputs/W7D1_EVALUATION.md

## Conclusion

The Haystack experiment successfully demonstrated document indexing, BM25 retrieval, dense retrieval, and manual comparison using the same set of 10 questions.
