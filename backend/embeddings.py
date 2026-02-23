"""OpenAI embeddings for chunks and queries."""
from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL

_client: OpenAI | None = None


def _client_or_raise() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a list of texts with OpenAI. Batches long lists."""
    if not texts:
        return []
    client = _client_or_raise()
    model = model or EMBEDDING_MODEL
    out = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = [t[:8000] for t in texts[i : i + batch_size]]
        r = client.embeddings.create(input=batch, model=model)
        for e in r.data:
            out.append(e.embedding)
    return out


def embed_single(text: str, model: str | None = None) -> list[float]:
    """Embed a single string."""
    return embed_texts([text], model)[0]

