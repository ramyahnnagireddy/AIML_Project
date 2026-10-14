
"""Configuration for the multi-document RAG project."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_MODEL = "llama3.2:3b"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"

TOP_K = 3
MLFLOW_EXPERIMENT_NAME = "W7D5_Multi_Document_RAG"
