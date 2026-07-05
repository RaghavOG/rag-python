from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any


class OllamaProvider:
    """Ollama local provider (chat + embeddings) via HTTP.

    Defaults to `http://localhost:11434`.
    """

    def __init__(self, *, base_url: str = "http://localhost:11434") -> None:
        try:
            import requests
        except ImportError as e:
            raise RuntimeError("Ollama provider requires `pip install requests`") from e
        self._requests = requests
        self._base_url = base_url.rstrip("/")

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
            raise RuntimeError("OllamaProvider requires `model=...` (e.g. llama3.1)")
        payload: dict[str, Any] = {
            "model": model,
            "messages": [],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["messages"].append({"role": "system", "content": system})
        payload["messages"].append({"role": "user", "content": user})

        r = self._requests.post(f"{self._base_url}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        msg = (data.get("message") or {}).get("content")
        return (msg or "").strip()

    def generate_stream(
        self,
        *,
        user: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        if not model:
            raise RuntimeError("OllamaProvider requires `model=...` (e.g. llama3.1)")
        payload: dict[str, Any] = {
            "model": model,
            "messages": [],
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["messages"].append({"role": "system", "content": system})
        payload["messages"].append({"role": "user", "content": user})

        with self._requests.post(
            f"{self._base_url}/api/chat", json=payload, timeout=120, stream=True
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                data = json.loads(line)
                msg = (data.get("message") or {}).get("content")
                if msg:
                    yield msg
                if data.get("done"):
                    break

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        if not model:
            raise RuntimeError("OllamaProvider.embed requires embedding `model=...` (e.g. mxbai-embed-large)")
        payload = {"model": model, "input": texts}
        r = self._requests.post(f"{self._base_url}/api/embed", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        embs = data.get("embeddings") or []
        return embs

