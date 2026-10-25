
"""Configuration for the multi-document RAG project."""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

# Override this value for Docker or a deployed environment.
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

OLLAMA_CHAT_MODEL = os.getenv(
    "OLLAMA_CHAT_MODEL",
    "llama3.2:3b",
)
OLLAMA_EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "nomic-embed-text:latest",
)

TOP_K = int(os.getenv("RAG_TOP_K", "3"))
MLFLOW_EXPERIMENT_NAME = "W7D5_Multi_Document_RAG"
