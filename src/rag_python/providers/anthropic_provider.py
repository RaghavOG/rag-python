from __future__ import annotations


class AnthropicProvider:
    def __init__(self, *, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("Anthropic provider requires `pip install anthropic`") from e
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        *,
        user: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        if not model:
            raise RuntimeError("AnthropicProvider requires `model=...` (e.g. claude-...)")
        msg = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            messages=[{"role": "user", "content": user}],
        )
        # SDK returns a content list; join any text blocks.
        parts = []
        for block in getattr(msg, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return ("\n".join(parts)).strip()

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        raise RuntimeError("Anthropic does not provide embeddings in this package. Use OpenAI/Ollama or local embeddings.")

