"""LLM streaming helpers with graceful fallback to non-streaming generate."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def stream_generate(
    llm: Any,
    *,
    user: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Iterator[str]:
    """Yield text chunks from an LLM provider, falling back to a single chunk."""
    if hasattr(llm, "generate_stream"):
        yield from llm.generate_stream(
            user=user,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return
    text = llm.generate(
        user=user,
        system=system,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if text:
        yield text
