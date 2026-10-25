W7D1 RETRIEVAL EVALUATION
=========================

Comparison: BM25 Retriever vs Dense Retriever
Documents indexed: 5
Questions evaluated: 10

MANUAL EVALUATION
-----------------

Q1. What is machine learning?
BM25: machine_learning_basics.pdf ranked 2nd
Dense: machine_learning_basics.pdf ranked 1st
Observation: Dense retrieval provided the most relevant document at rank 1.

Q2. What is supervised learning?
BM25: machine_learning_basics.pdf ranked 2nd
Dense: machine_learning_basics.pdf ranked 1st
Observation: Dense retrieval ranked the relevant document higher.

Q3. What is deep learning?
BM25: ai_ml_reference.pdf ranked 1st
Dense: ai_ml_reference.pdf ranked 1st
Observation: Both methods identified the same relevant document at rank 1.

Q4. What is natural language processing?
BM25: nlp_basics.pdf ranked 1st
Dense: nlp_basics.pdf ranked 1st
Observation: Both methods produced the same relevant document at rank 1.

Q5. What is a neural network?
BM25: information_retrieval.pdf ranked 1st
Dense: ai_ml_reference.pdf ranked 1st
Observation: Dense retrieval returned the AI/ML reference document first, while BM25 returned an information-retrieval document first. The results differ.

Q6. What is retrieval augmented generation?
BM25: sample_document.pdf ranked 1st
Dense: information_retrieval.pdf ranked 1st
Observation: Dense retrieval returned the document specifically covering information retrieval and RAG-related concepts.

Q7. What is model training?
BM25: ai_ml_reference.pdf ranked 1st
Dense: ai_ml_reference.pdf ranked 1st
Observation: Both methods identified the same document at rank 1.

Q8. What is classification?
BM25: ai_ml_reference.pdf ranked 1st
Dense: machine_learning_basics.pdf ranked 1st
Observation: The two retrieval methods produced different top-ranked documents.

Q9. What is an embedding?
BM25: sample_document.pdf ranked 1st
Dense: sample_document.pdf ranked 1st
Observation: Both methods identified the same document at rank 1.

Q10. What is the purpose of evaluation in machine learning?
BM25: sample_document.pdf ranked 1st
Dense: ai_ml_reference.pdf ranked 1st
Observation: The two retrieval methods produced different top-ranked documents.

SUMMARY
-------
BM25 uses lexical term matching and produced strong results when the query terms closely matched document text.

Dense retrieval uses vector embeddings and semantic similarity, allowing conceptually related content to be retrieved even when exact wording differs.

Across the 10 questions, several queries produced the same top-ranked document with both methods, while other queries produced different rankings.

The experiment demonstrates that retrieval performance depends on the query and document content. BM25 and dense retrieval provide complementary retrieval approaches.

Evidence source:
outputs/retrieval_results.json
