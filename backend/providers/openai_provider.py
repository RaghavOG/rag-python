from __future__ import annotations

from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL


class OpenAIProvider:
    def __init__(self, *, api_key: str | None = None) -> None:
        api_key = api_key or OPENAI_API_KEY
        if not api_key:
            raise RuntimeError("OpenAI API key missing. Set OPENAI_API_KEY or pass api_key=...")
        self._client = OpenAI(api_key=api_key)

    def generate(
        self,
        *,
        user: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        r = self._client.chat.completions.create(
            model=model or LLM_MODEL,
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = [t[:8000] for t in texts[i : i + batch_size]]
            r = self._client.embeddings.create(input=batch, model=model or EMBEDDING_MODEL)
            for e in r.data:
                out.append(e.embedding)
        return out

