from __future__ import annotations

from collections.abc import Iterator
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

    def generate_stream(
        self,
        *,
        user: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Iterator[str]: ...


class EmbeddingProvider(Protocol):
    """Minimal embeddings interface for this RAG system."""

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]: ...

