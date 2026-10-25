import os
from pathlib import Path

import chromadb
import ollama


# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

# Reuse the PDF created during W5D4
SOURCE_DIR = BASE_DIR.parent / "W5D4_ChromaDB_Semantic_Search"
PDF_PATH = SOURCE_DIR / "sample_document.pdf"

# Keep W5D5's vector database separate from previous days
CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "llama3.2:3b"
COLLECTION_NAME = "w5d5_qa_documents"


# -----------------------------
# Sample knowledge document
# -----------------------------
DEFAULT_DOCUMENT = """
Artificial Intelligence (AI) is the field of creating systems that can
perform tasks that normally require human intelligence.

Machine Learning (ML) is a subset of AI in which computer systems learn
patterns from data and use those patterns to make predictions or decisions.

Deep Learning is a subset of machine learning that uses neural networks
with multiple layers. It is commonly used for image classification,
natural language processing, speech recognition, and other complex tasks.

Large Language Models (LLMs) are deep learning models trained on large
amounts of text. They can generate and understand natural language.

Retrieval-Augmented Generation (RAG) combines information retrieval with
language generation. A user's question is first used to retrieve relevant
documents or passages. The retrieved context is then provided to a language
model so that it can generate a grounded answer.

ChromaDB is a vector database that can store documents and embeddings and
perform similarity search. It is useful for retrieving relevant information
for RAG applications.

Ollama allows large language models such as Llama to run locally on a
computer. Local inference can provide a simple way to experiment with LLMs
without sending prompts to a remote model service.
"""


# -----------------------------
# Create / load ChromaDB
# -----------------------------
def get_collection():
    """Create a persistent ChromaDB collection and add knowledge data."""

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "W5D5 Local Q&A Bot knowledge base"},
    )

    # Add the document only when the collection is empty.
    if collection.count() == 0:
        collection.add(
            ids=["ai_ml_rag_001"],
            documents=[DEFAULT_DOCUMENT],
        )

    return collection


# -----------------------------
# Similarity search
# -----------------------------
def retrieve_context(collection, question, n_results=1):
    """Retrieve relevant context from ChromaDB."""

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )

    documents = results.get("documents", [[]])[0]

    if not documents:
        return "No relevant context was found in the knowledge base."

    return "\n\n".join(documents)


# -----------------------------
# Ollama answer generation
# -----------------------------
def generate_answer(question, context):
    """Generate an answer using the local Ollama model."""

    system_prompt = """
You are a helpful AI/ML mentor for beginners.

Answer the user's question using the supplied context.
Keep the explanation clear and concise.

If the answer is not supported by the context, say:
"I don't have enough information in the provided knowledge base."

Do not invent facts.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}\n\n"
                    "Answer based on the context:"
                ),
            },
        ],
    )

    return response["message"]["content"].strip()


# -----------------------------
# Q&A function
# -----------------------------
def answer_question(collection, question):
    """Retrieve context and generate an answer."""

    context = retrieve_context(collection, question)
    answer = generate_answer(question, context)

    return context, answer


# -----------------------------
# Main test
# -----------------------------
def main():
    print("=" * 70)
    print("W5D5 - LOCAL Q&A BOT")
    print("=" * 70)
    print(f"Ollama model: {MODEL_NAME}")
    print(f"ChromaDB path: {CHROMA_DIR}")
    print(f"Source PDF available: {PDF_PATH.exists()}")

    collection = get_collection()

    print(f"ChromaDB documents: {collection.count()}")

    questions = [
        "What is Machine Learning?",
        "What is Retrieval-Augmented Generation?",
        "What is ChromaDB?",
        "What is Ollama used for?",
        "What is the difference between AI and Machine Learning?",
    ]

    output_lines = []

    for index, question in enumerate(questions, start=1):
        print(f"\nQuestion {index}: {question}")

        context, answer = answer_question(collection, question)

        print(f"Answer: {answer}")

        output_lines.append(f"QUESTION {index}")
        output_lines.append(question)
        output_lines.append("")
        output_lines.append("RETRIEVED CONTEXT")
        output_lines.append(context)
        output_lines.append("")
        output_lines.append("OLLAMA ANSWER")
        output_lines.append(answer)
        output_lines.append("")
        output_lines.append("-" * 70)

    output_path = BASE_DIR / "outputs" / "local_qa_output.txt"
    output_path.write_text("\n".join(output_lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"Output evidence saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    