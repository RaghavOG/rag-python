"""Query understanding & rewriting: expand/rewrite user query for better retrieval."""
from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL, MULTI_QUERY_N

_client: OpenAI | None = None


def _client_or_raise() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def rewrite_for_retrieval(query: str, n_queries: int = 3) -> list[str]:
    """Generate n alternative queries (multi-query) for better recall."""
    if n_queries <= 1:
        return [query]
    client = _client_or_raise()
    prompt = (
        "You are a search expert. Given the user question below, generate "
        f"{n_queries} different search queries that would help find relevant documents. "
        "Output ONLY the queries, one per line, no numbering or bullets."
    )
    try:
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        text = (r.choices[0].message.content or "").strip()
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


def understand_query(query: str) -> str:
    """Optional: rewrite single query for clarity (e.g. spell correction, intent)."""
    client = _client_or_raise()
    try:
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite the user's question to be a clear, standalone search query. "
                               "Preserve intent. Output only the rewritten query.",
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=150,
        )
        out = (r.choices[0].message.content or "").strip().strip('"')
        return out if out else query
    except Exception:
        return query

