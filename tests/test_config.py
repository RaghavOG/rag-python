from pathlib import Path

from rag_python.config import (
    CHUNK_STRATEGY,
    DATA_DIR,
    CHROMA_PERSIST_DIR,
    GUARDRAILS_ENABLED,
    LLM_MODEL,
    EMBEDDING_MODEL,
)


def test_config_defaults():
    assert LLM_MODEL
    assert EMBEDDING_MODEL
    assert CHUNK_STRATEGY in ("recursive", "structure_aware", "semantic")
    assert isinstance(GUARDRAILS_ENABLED, bool)
    assert isinstance(DATA_DIR, Path)
    assert isinstance(CHROMA_PERSIST_DIR, Path)
