"""Full RAG pipeline: Query → Understanding/Rewrite → Retrieval (multi-query) → Rerank → LLM → Guardrails → Eval/Retry."""
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .cleaning import clean_document
from .chunking import chunk_text, Chunk
from .document_loaders import load_file, load_directory, LoadedDocument
from .vector_store import ingest_chunks, delete_all
from .retrieval import retrieve as rag_retrieve
from .generation import generate, generate_stream
from .guardrails import check_prompt_injection, check_hallucination
from .evaluation import evaluate_rag, should_retry, self_correct
from .providers import LLMProvider, EmbeddingProvider, make_llm_provider, make_embedding_provider
from .config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_STRATEGY
from .options import QueryConfig, SearchConfig
from .log import get_logger

logger = get_logger("pipeline")


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    evaluation: dict
    retried: bool


class RAGStream:
    """Streaming query result. Iterate for tokens; read ``result`` after exhaustion."""

    def __init__(
        self,
        question: str,
        *,
        search: SearchConfig,
        query_config: QueryConfig,
        multi_query: bool | None,
        llm_model: str | None,
        embedding_model: str | None,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
    ) -> None:
        self._question = question
        self._search = search
        self._query_config = query_config
        self._multi_query = multi_query
        self._llm_model = llm_model
        self._embedding_model = embedding_model
        self._llm = llm
        self._embedder = embedder
        self._result: RAGResponse | None = None
        self._started = False

    def __iter__(self) -> Iterator[str]:
        if self._started:
            return iter(())
        self._started = True

        ctx = _prepare_query_context(
            self._question,
            search=self._search,
            query_config=self._query_config,
            multi_query=self._multi_query,
            llm_model=self._llm_model,
            embedding_model=self._embedding_model,
            llm=self._llm,
            embedder=self._embedder,
        )
        if ctx.blocked is not None:
            self._result = ctx.blocked
            yield ctx.blocked.answer
            return
        if not ctx.context_chunks:
            self._result = RAGResponse(
                answer="No relevant documents found for your question.",
                sources=[],
                evaluation={"faithfulness": 1.0, "relevance": 0.0},
                retried=False,
            )
            yield self._result.answer
            return

        parts: list[str] = []
        for token in generate_stream(
            ctx.question,
            ctx.context_chunks,
            model=self._llm_model,
            llm=self._llm,
        ):
            parts.append(token)
            yield token

        answer = "".join(parts)
        self._result = _finalize_response(
            ctx.question,
            answer,
            ctx.context_chunks,
            ctx.context_str,
            ctx.sources,
            query_config=self._query_config,
            llm_model=self._llm_model,
            llm=self._llm,
        )
        logger.info(
            "query_stream_complete chunks=%d faithfulness=%.2f relevance=%.2f",
            len(ctx.context_chunks),
            self._result.evaluation.get("faithfulness", 0.0),
            self._result.evaluation.get("relevance", 0.0),
        )

    @property
    def result(self) -> RAGResponse:
        if self._result is None:
            raise RuntimeError("Stream not finished; consume the iterator first")
        return self._result


@dataclass
class _QueryContext:
    question: str
    context_chunks: list[str]
    context_str: str
    sources: list[dict]
    blocked: RAGResponse | None = None


def _embed_fn_for_semantic(embedder: EmbeddingProvider, embedding_model: str | None):
    def _fn(texts: list[str]):
        return embedder.embed(texts, model=embedding_model)

    return _fn


def _load_documents(
    paths: list[Path] | None = None,
    data_path: Path | None = None,
    *,
    extensions: tuple[str, ...] = (".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".html"),
) -> list[LoadedDocument]:
    """Load documents from explicit paths and/or a data directory."""
    docs: list[LoadedDocument] = []
    seen_sources: set[str] = set()

    if paths:
        for raw in paths:
            path = Path(raw)
            if path.is_file():
                doc = load_file(path)
                if doc and doc.content.strip() and doc.source not in seen_sources:
                    seen_sources.add(doc.source)
                    docs.append(doc)
            elif path.is_dir():
                for doc in load_directory(path, extensions=extensions):
                    if doc.source not in seen_sources:
                        seen_sources.add(doc.source)
                        docs.append(doc)

    if data_path is not None:
        root = Path(data_path)
        if root.is_file():
            doc = load_file(root)
            if doc and doc.content.strip() and doc.source not in seen_sources:
                docs.append(doc)
        elif root.is_dir():
            for doc in load_directory(root, extensions=extensions):
                if doc.source not in seen_sources:
                    seen_sources.add(doc.source)
                    docs.append(doc)

    return docs


def _ingest_documents(
    docs: list[LoadedDocument],
    *,
    clean: bool,
    chunk_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str | None,
    embedder: EmbeddingProvider,
) -> int:
    all_chunks: list[Chunk] = []
    for doc in docs:
        text = doc.content
        if clean:
            text = clean_document(text)
        if chunk_strategy == "semantic":
            chunks = chunk_text(
                text,
                strategy="semantic",
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                metadata=doc.metadata,
                embed_fn=_embed_fn_for_semantic(embedder, embedding_model),
            )
        else:
            chunks = chunk_text(
                text,
                strategy=chunk_strategy,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                metadata=doc.metadata,
            )
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    texts = [c.text for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]
    ingest_chunks(ids, texts, metadatas, embedder=embedder, embed_model=embedding_model)
    logger.info("ingest_complete documents=%d chunks=%d strategy=%s", len(docs), len(all_chunks), chunk_strategy)
    return len(all_chunks)


def ingest(
    data_path: Path | str | None = None,
    *,
    paths: list[Path | str] | None = None,
    clean: bool = True,
    chunk_strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    extensions: tuple[str, ...] | None = None,
    reindex: bool = False,
    embedding_model: str | None = None,
    embedder: EmbeddingProvider | None = None,
) -> int:
    """
    Load documents → clean → chunk → embed → vector store.
    Pass ``paths`` for multiple files/folders, or ``data_path`` for a single root.
    Returns number of chunks ingested.
    """
    if reindex:
        delete_all()
        logger.info("ingest_reindex cleared vector store")
    strategy = chunk_strategy or CHUNK_STRATEGY
    size = chunk_size or CHUNK_SIZE
    overlap = chunk_overlap or CHUNK_OVERLAP
    ext = extensions or (".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".html")
    embedder = embedder or make_embedding_provider("openai")

    path_list = [Path(p) for p in paths] if paths else None
    root = Path(data_path) if data_path else (None if path_list else Path(DATA_DIR))
    docs = _load_documents(path_list, root, extensions=ext)
    logger.info("ingest_loaded documents=%d extensions=%s", len(docs), ext)
    return _ingest_documents(
        docs,
        clean=clean,
        chunk_strategy=strategy,
        chunk_size=size,
        chunk_overlap=overlap,
        embedding_model=embedding_model,
        embedder=embedder,
    )


def _prepare_query_context(
    question: str,
    *,
    search: SearchConfig,
    query_config: QueryConfig,
    multi_query: bool | None,
    llm_model: str | None,
    embedding_model: str | None,
    llm: LLMProvider,
    embedder: EmbeddingProvider,
) -> _QueryContext:
    if query_config.use_guardrails:
        safe, msg = check_prompt_injection(question, llm=llm, llm_model=llm_model)
        if not safe:
            logger.warning("query_blocked reason=prompt_injection")
            return _QueryContext(
                question=question,
                context_chunks=[],
                context_str="",
                sources=[],
                blocked=RAGResponse(
                    answer=f"Request blocked: {msg}",
                    sources=[],
                    evaluation={},
                    retried=False,
                ),
            )

    hits = rag_retrieve(
        question,
        retriever=search.retriever,
        multi_query=multi_query,
        n_queries=search.multi_query_n,
        top_k_retrieve=search.top_k_retrieve,
        top_k_rerank=search.top_k_rerank,
        rerank_enabled=search.rerank_enabled,
        metadata_filter=search.metadata_filter,
        embedder=embedder,
        embedding_model=embedding_model,
        llm=llm,
        llm_model=llm_model,
    )
    logger.info(
        "retrieve_complete retriever=%s chunks=%d metadata_filter=%s",
        search.retriever,
        len(hits),
        bool(search.metadata_filter),
    )
    context_chunks = [h[0] for h in hits]
    sources = [{"text": h[0][:200], "metadata": h[1], "score": h[2]} for h in hits]
    context_str = "\n\n".join(context_chunks)
    return _QueryContext(
        question=question,
        context_chunks=context_chunks,
        context_str=context_str,
        sources=sources,
    )


def _finalize_response(
    question: str,
    answer: str,
    context_chunks: list[str],
    context_str: str,
    sources: list[dict],
    *,
    query_config: QueryConfig,
    llm_model: str | None,
    llm: LLMProvider,
    use_retry: bool | None = None,
    eval_threshold: float | None = None,
    max_retries: int | None = None,
) -> RAGResponse:
    use_guardrails = query_config.use_guardrails
    use_retry = query_config.use_retry if use_retry is None else use_retry
    eval_threshold = query_config.eval_threshold if eval_threshold is None else eval_threshold
    max_retries = query_config.max_retries if max_retries is None else max_retries
    retried = False

    if use_guardrails:
        grounded, msg = check_hallucination(answer, context_str, llm=llm, llm_model=llm_model)
        if not grounded:
            logger.warning("hallucination_detected message=%s", msg[:80])
            answer = answer + f"\n\n[Note: {msg}]"

    eval_scores = evaluate_rag(question, answer, context_str, llm=llm, llm_model=llm_model)

    if use_retry and should_retry(eval_scores["faithfulness"], eval_scores["relevance"], eval_threshold):
        logger.info("query_retry faithfulness=%.2f relevance=%.2f", eval_scores["faithfulness"], eval_scores["relevance"])
        for _ in range(max(0, max_retries - 1)):
            answer = self_correct(question, answer, context_str, llm=llm, llm_model=llm_model)
            eval_scores = evaluate_rag(question, answer, context_str, llm=llm, llm_model=llm_model)
            retried = True
            if not should_retry(eval_scores["faithfulness"], eval_scores["relevance"], eval_threshold):
                break

    logger.info(
        "query_complete chunks=%d faithfulness=%.2f relevance=%.2f retried=%s",
        len(context_chunks),
        eval_scores.get("faithfulness", 0.0),
        eval_scores.get("relevance", 0.0),
        retried,
    )
    return RAGResponse(answer=answer, sources=sources, evaluation=eval_scores, retried=retried)


def query(
    question: str,
    *,
    search: SearchConfig | None = None,
    query_config: QueryConfig | None = None,
    multi_query: bool | None = None,
    use_guardrails: bool | None = None,
    use_retry: bool | None = None,
    eval_threshold: float | None = None,
    max_retries: int | None = None,
    llm_model: str | None = None,
    embedding_model: str | None = None,
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
) -> RAGResponse:
    """
    Run full pipeline: input guardrails → retrieval → rerank → generate →
    output guardrails → evaluation → retry/self-correction if needed.
    """
    search_cfg = search or SearchConfig()
    q_cfg = query_config or QueryConfig()
    use_retry = q_cfg.use_retry if use_retry is None else use_retry
    eval_threshold = q_cfg.eval_threshold if eval_threshold is None else eval_threshold
    max_retries = q_cfg.max_retries if max_retries is None else max_retries

    llm = llm or make_llm_provider("openai")
    embedder = embedder or make_embedding_provider("openai")

    ctx = _prepare_query_context(
        question,
        search=search_cfg,
        query_config=q_cfg,
        multi_query=multi_query,
        llm_model=llm_model,
        embedding_model=embedding_model,
        llm=llm,
        embedder=embedder,
    )
    if ctx.blocked is not None:
        return ctx.blocked

    if not ctx.context_chunks:
        return RAGResponse(
            answer="No relevant documents found for your question.",
            sources=[],
            evaluation={"faithfulness": 1.0, "relevance": 0.0},
            retried=False,
        )

    answer = generate(ctx.question, ctx.context_chunks, model=llm_model, llm=llm)
    return _finalize_response(
        ctx.question,
        answer,
        ctx.context_chunks,
        ctx.context_str,
        ctx.sources,
        query_config=q_cfg,
        llm_model=llm_model,
        llm=llm,
        use_retry=use_retry,
        eval_threshold=eval_threshold,
        max_retries=max_retries,
    )


def query_stream(
    question: str,
    *,
    search: SearchConfig | None = None,
    query_config: QueryConfig | None = None,
    multi_query: bool | None = None,
    llm_model: str | None = None,
    embedding_model: str | None = None,
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
) -> RAGStream:
    """Stream answer tokens after retrieval. Access ``stream.result`` after iteration."""
    search_cfg = search or SearchConfig()
    q_cfg = query_config or QueryConfig()
    llm = llm or make_llm_provider("openai")
    embedder = embedder or make_embedding_provider("openai")
    logger.info("query_stream_start retriever=%s", search_cfg.retriever)
    return RAGStream(
        question,
        search=search_cfg,
        query_config=q_cfg,
        multi_query=multi_query,
        llm_model=llm_model,
        embedding_model=embedding_model,
        llm=llm,
        embedder=embedder,
    )

