# W7D3 Experiment Summary: LlamaIndex Document Querying

## 1. Objective

Implement document querying using LlamaIndex, evaluate retrieval and answer generation across ten questions, integrate persistent ChromaDB storage, and compare query latency between an in-memory vector index and a ChromaDB-backed vector index.

## 2. Environment and Technology

- Framework: LlamaIndex
- Vector database: ChromaDB
- Embedding model: `nomic-embed-text`
- Local language model: `qwen2.5:3b`
- Document format: Plain text (`.txt`)
- Documents: 5
- Evaluation queries: 10
- Retrieval setting: Top 3 source nodes

## 3. Implementation

### Experiment A: In-memory vector index

Script: `llamaindex_basic.py`

- Loaded five text documents.
- Generated embeddings using Ollama.
- Built an in-memory `VectorStoreIndex`.
- Queried the index with ten questions.
- Recorded generated answers, source nodes, and query latency.

### Experiment B: ChromaDB-backed vector index

Script: `llamaindex_chroma.py`

- Loaded the same five text documents.
- Created a persistent ChromaDB database.
- Stored document embeddings in the `w7d3_llamaindex_documents` collection.
- Queried the index using the same ten questions and model settings.
- Recorded generated answers, source nodes, and query latency.

## 4. Results

| Metric                |                    In-memory index |               ChromaDB-backed index |
| --------------------- | ---------------------------------: | ----------------------------------: |
| Documents loaded      |                                  5 |                                   5 |
| Queries attempted     |                                 10 |                                  10 |
| Successful queries    |                                 10 |                                  10 |
| Indexing time         |                     7.4446 seconds |                     10.4163 seconds |
| Average query latency |                    23.5307 seconds |                     24.7037 seconds |
| Results file          | `outputs/basic_query_results.json` | `outputs/chroma_query_results.json` |

_Latency measurements reflect this particular local run and may vary between runs._

## 5. Observations

1. Both implementations completed all ten queries successfully.
2. The in-memory index had an average query latency of approximately 23.53 seconds.
3. The ChromaDB-backed index had an average query latency of approximately 24.70 seconds.
4. The ChromaDB-backed experiment took longer to index the documents in this run.
5. ChromaDB provides persistent vector storage, allowing indexed data to be retained on disk rather than relying exclusively on an in-memory index.
6. Local language-model inference contributes substantially to the overall query time, so the observed latency difference should not be attributed to the vector database alone.

## 6. Conclusion

Both implementations successfully queried the five-document collection using ten questions. The in-memory experiment was faster in this run, while the ChromaDB experiment demonstrated integration with persistent vector storage. The saved JSON outputs provide evidence for further analysis of answers, source nodes, and individual query latency.

## 7. Output Evidence

- `outputs/basic_query_results.json`
- `outputs/chroma_query_results.json`
- `chroma_db/` — persistent ChromaDB storage directory

The `chroma_db/` directory contains local database files and should generally remain excluded from Git unless the internship requirements explicitly request committing database artifacts.
