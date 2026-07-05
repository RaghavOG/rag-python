"""LLM generation with context (RAG)."""
from .config import LLM_MODEL
from .providers import LLMProvider, make_llm_provider


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
    llm: LLMProvider | None = None,
) -> str:
    """Generate answer from query and retrieved context."""
    llm = llm or make_llm_provider("openai")
    context = "\n\n---\n\n".join(context_chunks)
    sys = system_prompt or RAG_SYSTEM
    try:
        return llm.generate(
            system=sys,
            user=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            model=model or LLM_MODEL,
            temperature=0.2,
            max_tokens=1024,
        )
    except Exception as e:
        return f"[Generation error: {e}]"

