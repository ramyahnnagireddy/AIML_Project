from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


OUTPUT_DIR = Path("documents")
OUTPUT_DIR.mkdir(exist_ok=True)

documents = {
    "machine_learning.pdf": {
        "title": "Machine Learning",
        "sections": [
            (
                "Supervised Learning",
                "Supervised learning uses labeled training data to learn a relationship "
                "between input features and target outputs. Classification predicts "
                "categories, while regression predicts continuous numerical values."
            ),
            (
                "Unsupervised Learning",
                "Unsupervised learning works with data without labeled target values. "
                "Clustering groups similar observations, while dimensionality reduction "
                "represents data using fewer features."
            ),
            (
                "Model Evaluation",
                "Common classification metrics include accuracy, precision, recall, "
                "and F1 score. Regression can be evaluated using mean squared error, "
                "mean absolute error, and R-squared."
            ),
        ],
    },
    "deep_learning.pdf": {
        "title": "Deep Learning",
        "sections": [
            (
                "Neural Networks",
                "A neural network contains layers of interconnected neurons. "
                "Weights are adjusted during training so that the network can learn "
                "patterns from input data."
            ),
            (
                "Convolutional Neural Networks",
                "Convolutional neural networks, or CNNs, are commonly used for image "
                "tasks. Convolution filters learn spatial patterns such as edges, "
                "textures, and shapes."
            ),
            (
                "Training",
                "Deep learning models are trained using an optimization algorithm "
                "such as gradient descent. Backpropagation calculates gradients that "
                "are used to update model parameters."
            ),
        ],
    },
    "natural_language_processing.pdf": {
        "title": "Natural Language Processing",
        "sections": [
            (
                "Text Processing",
                "Natural language processing, or NLP, enables computers to work with "
                "human language. Common preprocessing operations include tokenization, "
                "normalization, and removing unnecessary text."
            ),
            (
                "Embeddings",
                "Text embeddings represent words, sentences, or documents as numerical "
                "vectors. Similar meanings can be represented by vectors that are close "
                "to one another in an embedding space."
            ),
            (
                "Language Applications",
                "NLP is used for sentiment analysis, text classification, information "
                "retrieval, question answering, summarization, and conversational systems."
            ),
        ],
    },
    "computer_vision.pdf": {
        "title": "Computer Vision",
        "sections": [
            (
                "Image Classification",
                "Image classification assigns an image to one or more predefined "
                "categories. Neural networks can learn visual features automatically "
                "from labeled images."
            ),
            (
                "Object Detection",
                "Object detection identifies objects in an image and estimates their "
                "locations. Detection systems commonly produce bounding boxes and "
                "class labels."
            ),
            (
                "Image Segmentation",
                "Image segmentation divides an image into meaningful regions. "
                "Semantic segmentation assigns a class to pixels, while instance "
                "segmentation separates individual objects."
            ),
        ],
    },
    "mlops.pdf": {
        "title": "MLOps",
        "sections": [
            (
                "Machine Learning Operations",
                "MLOps applies software engineering and operations practices to "
                "machine learning systems. It helps teams develop, test, deploy, "
                "monitor, and maintain machine learning models."
            ),
            (
                "Experiment Tracking",
                "Experiment tracking records parameters, metrics, artifacts, and "
                "model information from machine learning experiments. MLflow is a "
                "commonly used platform for experiment tracking and model lifecycle "
                "management."
            ),
            (
                "Deployment and Monitoring",
                "Machine learning deployment makes a trained model available for "
                "applications or users. Monitoring can track prediction quality, "
                "latency, data changes, and model performance after deployment."
            ),
        ],
    },
}


def create_pdf(filename, title, sections):
    path = OUTPUT_DIR / filename

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title=title,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 20))

    for heading, text in sections:
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 12))

    doc.build(story)
    print(f"Created: {path}")


def main():
    for filename, data in documents.items():
        create_pdf(
            filename,
            data["title"],
            data["sections"],
        )

    print("\nAll 5 PDF documents created successfully.")


if __name__ == "__main__":
    main()
    