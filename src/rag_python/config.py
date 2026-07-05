"""Configuration loaded from environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# API keys (provider-specific)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Paths — default to current working directory (works when installed from PyPI)
PROJECT_ROOT = Path.cwd()
DATA_DIR = Path(os.getenv("RAG_PYTHON_DATA_DIR", PROJECT_ROOT / "data"))
CHROMA_PERSIST_DIR = Path(os.getenv("RAG_PYTHON_CHROMA_DIR", PROJECT_ROOT / "chroma_db"))

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "recursive")  # recursive | structure_aware | semantic

# Retrieval
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "20"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))
MULTI_QUERY_N = int(os.getenv("MULTI_QUERY_N", "3"))

# Guardrails
GUARDRAILS_ENABLED = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# Reranker (optional extra: pip install rag-python[rerank])
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
