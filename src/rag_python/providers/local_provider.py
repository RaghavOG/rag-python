"""Local sentence-transformers embeddings (no API key required)."""
from __future__ import annotations

import os

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class LocalEmbeddingProvider:
    """Offline embeddings via sentence-transformers."""

    def __init__(self, model_name: str | None = None) -> None:
        self.default_model = model_name or os.getenv("LOCAL_EMBEDDING_MODEL", _DEFAULT_MODEL)
        self._models: dict[str, object] = {}

    def _get_model(self, model_name: str):
        if model_name not in self._models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "Local embeddings require optional dependencies. "
                    "Install with: pip install rag-python[local]"
                ) from e
            self._models[model_name] = SentenceTransformer(model_name)
        return self._models[model_name]

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        model_name = model or self.default_model
        encoder = self._get_model(model_name)
        vectors = encoder.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]
