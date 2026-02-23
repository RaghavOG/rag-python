"""Query understanding & rewriting: expand/rewrite user query for better retrieval."""
from config import LLM_MODEL, MULTI_QUERY_N
from .providers import LLMProvider, make_llm_provider


def rewrite_for_retrieval(
    query: str,
    *,
    n_queries: int = 3,
    llm: LLMProvider | None = None,
    llm_model: str | None = None,
) -> list[str]:
    """Generate n alternative queries (multi-query) for better recall."""
    if n_queries <= 1:
        return [query]
    llm = llm or make_llm_provider("openai")
    prompt = (
        "You are a search expert. Given the user question below, generate "
        f"{n_queries} different search queries that would help find relevant documents. "
        "Output ONLY the queries, one per line, no numbering or bullets."
    )
    try:
        text = llm.generate(
            system=prompt,
            user=query,
            model=llm_model or LLM_MODEL,
            temperature=0.3,
            max_tokens=200,
        )
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        cleaned = []
        for line in lines[:n_queries]:
            for sep in (". ", ") ", "- ", " ", ""):
                if line.startswith(sep):
                    line = line[len(sep):].strip()
                if line and not line[0].isdigit():
                    break
            if line:
                cleaned.append(line)
        if not cleaned:
            return [query]
        return cleaned[:n_queries]
    except Exception:
        return [query]


def understand_query(query: str, *, llm: LLMProvider | None = None, llm_model: str | None = None) -> str:
    """Optional: rewrite single query for clarity (e.g. spell correction, intent)."""
    llm = llm or make_llm_provider("openai")
    try:
        out = llm.generate(
            system=(
                "Rewrite the user's question to be a clear, standalone search query. "
                "Preserve intent. Output only the rewritten query."
            ),
            user=query,
            model=llm_model or LLM_MODEL,
            temperature=0,
            max_tokens=150,
        )
        out = (out or "").strip().strip('"')
        return out if out else query
    except Exception:
        return query

