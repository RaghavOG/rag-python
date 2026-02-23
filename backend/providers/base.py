from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    """Minimal chat-capable LLM interface for this RAG system."""

    def generate(
        self,
        *,
        user: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str: ...


class EmbeddingProvider(Protocol):
    """Minimal embeddings interface for this RAG system."""

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]: ...

