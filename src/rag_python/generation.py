"""LLM generation with context (RAG)."""
from collections.abc import Iterator

from .config import LLM_MODEL
from .providers import LLMProvider, make_llm_provider
from .providers.streaming import stream_generate


RAG_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
    "If the context does not contain enough information, say so. Do not invent facts. "
    "Quote or paraphrase from the context when relevant."
)


def _build_user_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def generate(
    query: str,
    context_chunks: list[str],
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    llm: LLMProvider | None = None,
) -> str:
    """Generate answer from query and retrieved context."""
    llm = llm or make_llm_provider("openai")
    sys = system_prompt or RAG_SYSTEM
    try:
        return llm.generate(
            system=sys,
            user=_build_user_prompt(query, context_chunks),
            model=model or LLM_MODEL,
            temperature=0.2,
            max_tokens=1024,
        )
    except Exception as e:
        return f"[Generation error: {e}]"


def generate_stream(
    query: str,
    context_chunks: list[str],
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    llm: LLMProvider | None = None,
) -> Iterator[str]:
    """Stream answer tokens from query and retrieved context."""
    llm = llm or make_llm_provider("openai")
    sys = system_prompt or RAG_SYSTEM
    try:
        yield from stream_generate(
            llm,
            system=sys,
            user=_build_user_prompt(query, context_chunks),
            model=model or LLM_MODEL,
            temperature=0.2,
            max_tokens=1024,
        )
    except Exception as e:
        yield f"[Generation error: {e}]"
