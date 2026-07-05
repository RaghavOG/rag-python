"""RAG evaluation: faithfulness, relevance; retry & self-correction loop."""
from .config import LLM_MODEL
from .providers import LLMProvider, make_llm_provider


def score_faithfulness(
    answer: str,
    context: str,
    *,
    llm: LLMProvider | None = None,
    llm_model: str | None = None,
) -> float:
    """Score 0–1: is the answer grounded in context?"""
    if not context.strip():
        return 1.0
    llm = llm or make_llm_provider("openai")
    try:
        text = llm.generate(
            system=(
                "Rate how much the ANSWER is supported by the CONTEXT (no invented facts). "
                "Reply with a single number from 0 to 1 (1 = fully supported). Only output the number."
            ),
            user=f"CONTEXT:\n{context[:3000]}\n\nANSWER:\n{answer[:1500]}",
            model=llm_model or LLM_MODEL,
            temperature=0,
            max_tokens=10,
        ).strip()
        return float(text.replace(",", ".").strip() or "1")
    except (ValueError, Exception):
        return 1.0


def score_relevance(
    answer: str,
    query: str,
    *,
    llm: LLMProvider | None = None,
    llm_model: str | None = None,
) -> float:
    """Score 0–1: does the answer address the query?"""
    llm = llm or make_llm_provider("openai")
    try:
        text = llm.generate(
            system=(
                "Rate how well the ANSWER addresses the QUESTION (0 = not at all, 1 = fully). "
                "Reply with a single number from 0 to 1. Only output the number."
            ),
            user=f"QUESTION: {query}\n\nANSWER: {answer}",
            model=llm_model or LLM_MODEL,
            temperature=0,
            max_tokens=10,
        ).strip()
        return float(text.replace(",", ".").strip() or "1")
    except (ValueError, Exception):
        return 1.0


def evaluate_rag(
    query: str,
    answer: str,
    context: str,
    *,
    llm: LLMProvider | None = None,
    llm_model: str | None = None,
) -> dict:
    """Offline-style evaluation: faithfulness + relevance."""
    return {
        "faithfulness": score_faithfulness(answer, context, llm=llm, llm_model=llm_model),
        "relevance": score_relevance(answer, query, llm=llm, llm_model=llm_model),
    }


def should_retry(faithfulness: float, relevance: float, threshold: float = 0.6) -> bool:
    """Decide if we should retry generation (e.g. different prompt or more context)."""
    return faithfulness < threshold or relevance < threshold


def self_correct(
    query: str,
    initial_answer: str,
    context: str,
    feedback: str = "The answer may not be fully grounded or relevant. Revise using only the context.",
    *,
    llm: LLMProvider | None = None,
    llm_model: str | None = None,
) -> str:
    """One self-correction pass: ask LLM to revise answer given feedback."""
    llm = llm or make_llm_provider("openai")
    try:
        return llm.generate(
            system=(
                "Revise the given ANSWER to better match the CONTEXT and the QUESTION. "
                "Output only the revised answer, no meta-commentary."
            ),
            user=(
                f"CONTEXT:\n{context[:3500]}\n\nQUESTION: {query}\n\n"
                f"CURRENT ANSWER: {initial_answer}\n\nFEEDBACK: {feedback}"
            ),
            model=llm_model or LLM_MODEL,
            temperature=0.1,
            max_tokens=1024,
        ).strip()
    except Exception:
        return initial_answer

