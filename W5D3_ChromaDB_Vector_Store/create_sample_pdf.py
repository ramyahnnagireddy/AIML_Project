from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


PDF_PATH = "documents/ai_ml_reference.pdf"


content = [
    (
        "Introduction to Machine Learning",
        "Machine learning enables computers to learn patterns from data and use "
        "those patterns to make predictions or decisions."
    ),
    (
        "Supervised Learning",
        "Supervised learning uses labeled data. Classification predicts discrete "
        "categories, while regression predicts continuous numerical values."
    ),
    (
        "Model Training",
        "A machine learning model learns relationships from training data. "
        "The training process should be evaluated using data that was not used "
        "to train the model."
    ),
    (
        "Feature Scaling",
        "Feature scaling transforms numerical features to comparable ranges. "
        "It can improve the performance of algorithms such as support vector "
        "machines, k-nearest neighbors, and logistic regression."
    ),
    (
        "Hyperparameter Tuning",
        "Hyperparameter tuning searches for model settings that improve "
        "performance. Grid search and randomized search are common approaches."
    ),
    (
        "Cross-Validation",
        "Cross-validation divides available training data into multiple folds. "
        "It helps estimate how well a model may generalize to unseen data."
    ),
    (
        "Classification Metrics",
        "Precision measures how many predicted positive cases are actually "
        "positive. Recall measures how many actual positive cases are correctly "
        "identified. The confusion matrix summarizes classification results."
    ),
    (
        "Deep Learning",
        "Deep learning uses neural networks with multiple layers. These models "
        "can learn complex patterns from large datasets."
    ),
    (
        "Natural Language Processing",
        "Natural language processing, or NLP, enables computers to process and "
        "understand human language. NLP is used in applications such as "
        "question answering, text classification, and information retrieval."
    ),
    (
        "Vector Databases",
        "Vector databases store numerical representations called embeddings. "
        "They can retrieve documents based on semantic similarity between a "
        "query and stored document embeddings."
    ),
]


styles = getSampleStyleSheet()

document = SimpleDocTemplate(
    PDF_PATH,
    pagesize=A4,
    title="AI and Machine Learning Reference",
)

story = []

story.append(
    Paragraph(
        "AI and Machine Learning Reference",
        styles["Title"],
    )
)

story.append(Spacer(1, 20))

for heading, text in content:
    story.append(Paragraph(heading, styles["Heading2"]))
    story.append(Paragraph(text, styles["BodyText"]))
    story.append(Spacer(1, 12))

document.build(story)

print("Sample PDF created successfully.")
print(f"PDF path: {PDF_PATH}")
print(f"Number of sections: {len(content)}")
