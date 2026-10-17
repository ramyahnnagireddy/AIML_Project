import chromadb


# Connect to the existing persistent ChromaDB database
client = chromadb.PersistentClient(path="./chroma_db")

# Get the existing collection
collection = client.get_collection(name="aiml_documents")


print("=" * 60)
print("CHROMADB SIMILARITY SEARCH")
print("=" * 60)

print(f"Collection: {collection.name}")
print(f"Total documents: {collection.count()}")


# ---------------------------------------------------------
# 1. Similarity search using cosine distance
# ---------------------------------------------------------

query = "How can I improve the performance of a machine learning model?"

results = collection.query(
    query_texts=[query],
    n_results=3,
)

print("\n1. COSINE SIMILARITY SEARCH")
print("-" * 60)
print(f"Query: {query}")

for i, (doc, distance, metadata) in enumerate(
    zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ),
    start=1,
):
    print(f"\nResult {i}")
    print(f"Document: {doc}")
    print(f"Cosine distance: {distance:.4f}")
    print(f"Metadata: {metadata}")


# ---------------------------------------------------------
# 2. Metadata filtering
# ---------------------------------------------------------

filtered_results = collection.get(
    where={"category": "algorithm"},
    limit=5,
)

print("\n\n2. METADATA FILTERING")
print("-" * 60)
print("Filter: category = algorithm")

for i, (doc_id, doc, metadata) in enumerate(
    zip(
        filtered_results["ids"],
        filtered_results["documents"],
        filtered_results["metadatas"],
    ),
    start=1,
):
    print(f"\nResult {i}")
    print(f"ID: {doc_id}")
    print(f"Document: {doc}")
    print(f"Metadata: {metadata}")


# ---------------------------------------------------------
# 3. Combined query + metadata filter
# ---------------------------------------------------------

filtered_query = "Which algorithm is useful for classification?"

filtered_search = collection.query(
    query_texts=[filtered_query],
    n_results=3,
    where={"category": "algorithm"},
)

print("\n\n3. SIMILARITY SEARCH + METADATA FILTER")
print("-" * 60)
print(f"Query: {filtered_query}")
print("Filter: category = algorithm")

for i, (doc, distance, metadata) in enumerate(
    zip(
        filtered_search["documents"][0],
        filtered_search["distances"][0],
        filtered_search["metadatas"][0],
    ),
    start=1,
):
    print(f"\nResult {i}")
    print(f"Document: {doc}")
    print(f"Cosine distance: {distance:.4f}")
    print(f"Metadata: {metadata}")


print("\n" + "=" * 60)
print("SEARCH AND FILTERING TEST COMPLETED")
print("=" * 60)
