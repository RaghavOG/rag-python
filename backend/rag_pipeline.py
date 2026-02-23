"""Full RAG pipeline: Query → Understanding/Rewrite → Retrieval (multi-query) → Rerank → LLM → Guardrails → Eval/Retry."""
from dataclasses import dataclass
from pathlib import Path

from .cleaning import clean_document
from .chunking import chunk_text, Chunk
from .document_loaders import load_file, load_directory
from .embeddings import embed_texts
from .vector_store import ingest_chunks, delete_all
from .query_rewriting import understand_query, rewrite_for_retrieval
from .retrieval import retrieve as rag_retrieve
from .generation import generate
from .guardrails import check_prompt_injection, check_hallucination
from .evaluation import evaluate_rag, should_retry, self_correct
from config import (
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_STRATEGY,
    MAX_RETRIES,
)


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    evaluation: dict
    retried: bool


def _embed_fn_for_semantic(texts: list[str]):
    return embed_texts(texts)


def ingest(
    data_path: Path | str | None = None,
    *,
    clean: bool = True,
    chunk_strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    reindex: bool = False,
) -> int:
    """
    Load documents → clean → chunk → embed → vector store.
    Returns number of chunks ingested.
    """
    path = Path(data_path or DATA_DIR)
    if reindex:
        delete_all()
    strategy = chunk_strategy or CHUNK_STRATEGY
    size = chunk_size or CHUNK_SIZE
    overlap = chunk_overlap or CHUNK_OVERLAP

    if path.is_file():
        docs = [load_file(path)]
        docs = [d for d in docs if d is not None]
    else:
        docs = list(load_directory(path))

    all_chunks: list[Chunk] = []
    for doc in docs:
        text = doc.content
        if clean:
            text = clean_document(text)
        if strategy == "semantic":
            chunks = chunk_text(
                text,
                strategy="semantic",
                chunk_size=size,
                overlap=overlap,
                metadata=doc.metadata,
                embed_fn=_embed_fn_for_semantic,
            )
        else:
            chunks = chunk_text(
                text,
                strategy=strategy,
                chunk_size=size,
                overlap=overlap,
                metadata=doc.metadata,
            )
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    texts = [c.text for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]
    ingest_chunks(ids, texts, metadatas)
    return len(all_chunks)


def query(
    question: str,
    *,
    multi_query: bool = True,
    use_guardrails: bool = True,
    use_retry: bool = True,
    eval_threshold: float = 0.6,
) -> RAGResponse:
    """
    Run full pipeline: input guardrails → (optional) query understanding →
    multi-query retrieval → rerank → generate → output guardrails →
    evaluation → retry/self-correction if needed.
    """
    if use_guardrails:
        safe, msg = check_prompt_injection(question)
        if not safe:
            return RAGResponse(
                answer=f"Request blocked: {msg}",
                sources=[],
                evaluation={},
                retried=False,
            )

    # Optional single-query rewrite (kept as hook)
    # q_understood = understand_query(question)
    q_understood = question

    hits = rag_retrieve(q_understood, multi_query=multi_query)
    context_chunks = [h[0] for h in hits]
    sources = [{"text": h[0][:200], "metadata": h[1], "score": h[2]} for h in hits]
    context_str = "\n\n".join(context_chunks)

    if not context_chunks:
        return RAGResponse(
            answer="No relevant documents found for your question.",
            sources=[],
            evaluation={"faithfulness": 1.0, "relevance": 0.0},
            retried=False,
        )

    answer = generate(q_understood, context_chunks)
    retried = False

    if use_guardrails:
        grounded, msg = check_hallucination(answer, context_str)
        if not grounded:
            answer = answer + f"\n\n[Note: {msg}]"

    eval_scores = evaluate_rag(q_understood, answer, context_str)

    if use_retry and should_retry(eval_scores["faithfulness"], eval_scores["relevance"], eval_threshold):
        for _ in range(MAX_RETRIES - 1):
            answer = self_correct(q_understood, answer, context_str)
            eval_scores = evaluate_rag(q_understood, answer, context_str)
            retried = True
            if not should_retry(eval_scores["faithfulness"], eval_scores["relevance"], eval_threshold):
                break

    return RAGResponse(answer=answer, sources=sources, evaluation=eval_scores, retried=retried)

