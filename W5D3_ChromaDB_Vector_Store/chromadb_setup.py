import chromadb


# Create a persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get the collection
collection = client.get_or_create_collection(
    name="aiml_documents",
    configuration={"hnsw": {"space": "cosine"}},
)

# 20 sample AI/ML documents
documents = [
    "Machine learning enables computers to learn patterns from data.",
    "Supervised learning uses labeled data to train predictive models.",
    "Unsupervised learning discovers patterns in unlabeled data.",
    "Classification predicts discrete categories such as spam or not spam.",
    "Regression predicts continuous numerical values.",
    "Linear regression models the relationship between input and output variables.",
    "Logistic regression is commonly used for binary classification.",
    "Decision trees make predictions using a sequence of feature-based decisions.",
    "Random forests combine multiple decision trees to improve predictions.",
    "Support vector machines find decision boundaries between classes.",
    "K-nearest neighbors predicts using nearby training examples.",
    "Feature scaling can improve the performance of many machine learning algorithms.",
    "Cross-validation helps estimate how well a model generalizes to unseen data.",
    "Hyperparameter tuning searches for model settings that improve performance.",
    "Precision measures the proportion of predicted positives that are actually positive.",
    "Recall measures the proportion of actual positives correctly identified.",
    "A confusion matrix summarizes classification predictions.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables computers to work with human language.",
    "Vector databases store embeddings and support similarity-based retrieval.",
]

# Metadata for the 20 documents
metadatas = [
    {"topic": "machine_learning", "category": "fundamentals"},
    {"topic": "supervised_learning", "category": "learning_type"},
    {"topic": "unsupervised_learning", "category": "learning_type"},
    {"topic": "classification", "category": "task"},
    {"topic": "regression", "category": "task"},
    {"topic": "linear_regression", "category": "algorithm"},
    {"topic": "logistic_regression", "category": "algorithm"},
    {"topic": "decision_tree", "category": "algorithm"},
    {"topic": "random_forest", "category": "algorithm"},
    {"topic": "svm", "category": "algorithm"},
    {"topic": "knn", "category": "algorithm"},
    {"topic": "feature_scaling", "category": "preprocessing"},
    {"topic": "cross_validation", "category": "evaluation"},
    {"topic": "hyperparameter_tuning", "category": "optimization"},
    {"topic": "precision", "category": "metric"},
    {"topic": "recall", "category": "metric"},
    {"topic": "confusion_matrix", "category": "evaluation"},
    {"topic": "deep_learning", "category": "neural_networks"},
    {"topic": "nlp", "category": "ai"},
    {"topic": "vector_database", "category": "retrieval"},
]

# Add the 20 documents
ids = [f"doc_{i + 1}" for i in range(20)]

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
)

# Verify collection contents
print("ChromaDB setup completed successfully.")
print(f"Collection name: {collection.name}")
print(f"Number of documents: {collection.count()}")
print("Added documents: 20")
print("Embedding space: cosine")
