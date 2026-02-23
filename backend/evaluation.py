"""RAG evaluation: faithfulness, relevance; retry & self-correction loop."""
from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL, MAX_RETRIES

_client: OpenAI | None = None


def _client_or_raise() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def score_faithfulness(answer: str, context: str) -> float:
    """Score 0–1: is the answer grounded in context?"""
    if not context.strip():
        return 1.0
    client = _client_or_raise()
    try:
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Rate how much the ANSWER is supported by the CONTEXT (no invented facts). "
                               "Reply with a single number from 0 to 1 (1 = fully supported). Only output the number.",
                },
                {"role": "user", "content": f"CONTEXT:\n{context[:3000]}\n\nANSWER:\n{answer[:1500]}"},
            ],
            temperature=0,
            max_tokens=10,
        )
        text = (r.choices[0].message.content or "1").strip()
        return float(text.replace(",", ".").strip() or "1")
    except (ValueError, Exception):
        return 1.0


def score_relevance(answer: str, query: str) -> float:
    """Score 0–1: does the answer address the query?"""
    client = _client_or_raise()
    try:
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Rate how well the ANSWER addresses the QUESTION (0 = not at all, 1 = fully). "
                               "Reply with a single number from 0 to 1. Only output the number.",
                },
                {"role": "user", "content": f"QUESTION: {query}\n\nANSWER: {answer}"},
            ],
            temperature=0,
            max_tokens=10,
        )
        text = (r.choices[0].message.content or "1").strip()
        return float(text.replace(",", ".").strip() or "1")
    except (ValueError, Exception):
        return 1.0


def evaluate_rag(query: str, answer: str, context: str) -> dict:
    """Offline-style evaluation: faithfulness + relevance."""
    return {
        "faithfulness": score_faithfulness(answer, context),
        "relevance": score_relevance(answer, query),
    }


def should_retry(faithfulness: float, relevance: float, threshold: float = 0.6) -> bool:
    """Decide if we should retry generation (e.g. different prompt or more context)."""
    return faithfulness < threshold or relevance < threshold


def self_correct(
    query: str,
    initial_answer: str,
    context: str,
    feedback: str = "The answer may not be fully grounded or relevant. Revise using only the context.",
) -> str:
    """One self-correction pass: ask LLM to revise answer given feedback."""
    client = _client_or_raise()
    try:
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Revise the given ANSWER to better match the CONTEXT and the QUESTION. "
                               "Output only the revised answer, no meta-commentary.",
                },
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context[:3500]}\n\nQUESTION: {query}\n\n"
                               f"CURRENT ANSWER: {initial_answer}\n\nFEEDBACK: {feedback}",
                },
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return (r.choices[0].message.content or initial_answer).strip()
    except Exception:
        return initial_answer

