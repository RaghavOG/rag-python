from __future__ import annotations


class GeminiProvider:
    def __init__(self, *, api_key: str | None = None, vertexai: bool = False, project: str | None = None, location: str | None = None) -> None:
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError("Gemini provider requires `pip install google-genai`") from e
        self._genai = genai
        if vertexai:
            if not project or not location:
                raise RuntimeError("Vertex AI requires project=... and location=...")
            self._client = genai.Client(vertexai=True, project=project, location=location)
        else:
            if not api_key:
                raise RuntimeError("Gemini provider requires api_key=... (GEMINI_API_KEY)")
            self._client = genai.Client(api_key=api_key)

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
            raise RuntimeError("GeminiProvider requires `model=...` (e.g. gemini-2.0-flash)")
        # Minimal portability: flatten system + user into a single prompt.
        prompt = f"{system.strip()}\n\n{user}".strip() if system else user
        resp = self._client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = getattr(resp, "text", None)
        if text:
            return text.strip()
        # Fallback: try candidates
        return str(resp).strip()

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        raise RuntimeError("Gemini embeddings are not wired here yet. Use OpenAI/Ollama or local embeddings.")

