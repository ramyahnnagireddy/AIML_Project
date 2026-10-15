
from pathlib import Path

from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from pypdf import PdfReader

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"


def load_pdf_documents(data_dir=DATA_DIR):
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    documents = []
    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"[Page {page_number}]\n{text}")

        content = "\n\n".join(pages)
        if not content:
            raise ValueError(f"No extractable text in {pdf_path.name}")

        documents.append(
            Document(
                id=pdf_path.stem,
                content=content,
                meta={"source": pdf_path.name},
            )
        )

    return documents


def build_bm25_pipeline():
    documents = load_pdf_documents()
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)

    pipeline = Pipeline()
    pipeline.add_component(
        "retriever",
        InMemoryBM25Retriever(document_store=document_store),
    )
    return pipeline, document_store


def retrieve_bm25(pipeline, question, top_k=3):
    result = pipeline.run(
        {
            "retriever": {
                "query": question,
                "top_k": top_k,
            }
        }
    )

    return [
        {
            "source": doc.meta.get("source", "unknown"),
            "score": float(doc.score or 0.0),
            "content": doc.content or "",
        }
        for doc in result["retriever"]["documents"]
    ]
