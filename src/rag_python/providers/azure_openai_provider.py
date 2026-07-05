from __future__ import annotations

from collections.abc import Iterator

from openai import AzureOpenAI

from ..config import LLM_MODEL, EMBEDDING_MODEL


class AzureOpenAIProvider:
    """Azure OpenAI provider.

    Note: For Azure, `model` should be your *deployment name*.
    """

    def __init__(
        self,
        *,
        azure_endpoint: str,
        api_key: str,
        api_version: str = "2023-09-01-preview",
    ) -> None:
        if not azure_endpoint:
            raise RuntimeError("azure_endpoint is required for Azure OpenAI")
        if not api_key:
            raise RuntimeError("api_key is required for Azure OpenAI")
        self._client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )

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

    def generate_stream(
        self,
        *,
        user: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=model or LLM_MODEL,
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

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

