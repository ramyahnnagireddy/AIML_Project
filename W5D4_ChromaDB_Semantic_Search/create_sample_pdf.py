from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "sample_document.pdf"


def create_pdf():
    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
    )

    styles = getSampleStyleSheet()

    story = []

    title = Paragraph(
        "Introduction to Artificial Intelligence and Machine Learning",
        styles["Title"],
    )

    story.append(title)
    story.append(Spacer(1, 20))

    sections = [
        (
            "Artificial Intelligence",
            "Artificial Intelligence (AI) is a field of computer science "
            "focused on creating systems that can perform tasks that normally "
            "require human intelligence. AI systems can process information, "
            "recognize patterns, understand language, and support decision making.",
        ),
        (
            "Machine Learning",
            "Machine Learning (ML) is a branch of AI in which algorithms learn "
            "patterns from data. Instead of explicitly programming every rule, "
            "a machine learning model uses examples to learn a relationship "
            "between input features and expected outputs.",
        ),
        (
            "Supervised Learning",
            "Supervised learning uses labeled training data. Classification "
            "predicts categories such as spam or not spam, while regression "
            "predicts continuous values such as house prices.",
        ),
        (
            "Embeddings",
            "Text embeddings convert text into numerical vectors. Similar "
            "pieces of text tend to have vectors that are close together in "
            "the embedding space. Embeddings are useful for semantic search "
            "and retrieval systems.",
        ),
        (
            "Vector Databases",
            "Vector databases store numerical embeddings and make it possible "
            "to efficiently search for vectors that are similar to a query. "
            "ChromaDB is a vector database that can be used to store documents, "
            "embeddings, and metadata.",
        ),
        (
            "Retrieval Augmented Generation",
            "Retrieval Augmented Generation (RAG) combines information retrieval "
            "with a large language model. Relevant document chunks are retrieved "
            "from a vector database and provided to the language model as context. "
            "The model then generates an answer based on the retrieved information.",
        ),
        (
            "Ollama",
            "Ollama allows large language models to run locally. In this project, "
            "Ollama is used both for generating text embeddings and for generating "
            "an answer from retrieved document context.",
        ),
    ]

    for heading, content in sections:
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Paragraph(content, styles["BodyText"]))
        story.append(Spacer(1, 12))

    document.build(story)

    print(f"PDF created successfully: {PDF_PATH}")


if __name__ == "__main__":
    create_pdf()
    