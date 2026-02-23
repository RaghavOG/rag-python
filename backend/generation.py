"""LLM generation with context (RAG)."""
from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL

_client: OpenAI | None = None


def _client_or_raise() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


RAG_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
    "If the context does not contain enough information, say so. Do not invent facts. "
    "Quote or paraphrase from the context when relevant."
)


def generate(
    query: str,
    context_chunks: list[str],
    *,
    model: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Generate answer from query and retrieved context."""
    client = _client_or_raise()
    context = "\n\n---\n\n".join(context_chunks)
    sys = system_prompt or RAG_SYSTEM
    try:
        r = client.chat.completions.create(
            model=model or LLM_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[Generation error: {e}]"

