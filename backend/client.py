"""High-level RAG client API.

This wraps the full RAG pipeline behind a simple interface:

    from backend.client import RAG

    rag = RAG(llm_model="gpt-4o-mini", embedding_model="text-embedding-3-small")
    rag.ingest(["./docs", "README.md"])
    answer = rag.query("What is our leave policy?")
    print(answer.text)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from config import DATA_DIR, LLM_MODEL, EMBEDDING_MODEL
from .providers import make_llm_provider, make_embedding_provider
from .rag_pipeline import ingest as _ingest, query as _query, RAGResponse


@dataclass
class RAGAnswer:
    text: str
    sources: list[dict]
    evaluation: dict
    retried: bool


class RAG:
    """User-facing RAG client."""

    def __init__(
        self,
        *,
        llm_provider: str = "openai",
        llm_model: str | None = None,
        embedding_provider: str = "openai",
        embedding_model: str | None = None,
        data_dir: str | Path | None = None,
        # Provider kwargs (optional)
        openai_api_key: str | None = None,
        azure_endpoint: str | None = None,
        azure_api_key: str | None = None,
        azure_api_version: str | None = None,
        anthropic_api_key: str | None = None,
        gemini_api_key: str | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        self.llm_provider_name = llm_provider
        self.embedding_provider_name = embedding_provider
        self.llm_model = llm_model or LLM_MODEL
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.data_dir = Path(data_dir) if data_dir else Path(DATA_DIR)

        # Providers
        self.llm = make_llm_provider(
            llm_provider,  # type: ignore[arg-type]
            api_key=openai_api_key or anthropic_api_key or gemini_api_key or azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
            base_url=ollama_base_url,
        )
        self.embedder = make_embedding_provider(
            embedding_provider,  # type: ignore[arg-type]
            api_key=openai_api_key or azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
            base_url=ollama_base_url,
        )

    def ingest(self, paths: Iterable[str | Path], *, reindex: bool = False) -> int:
        """Ingest the given files/directories into the vector store."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        for p in paths:
            p = Path(p)
            if p.is_file():
                target = self.data_dir / p.name
                if str(p.resolve()) != str(target.resolve()):
                    target.write_bytes(p.read_bytes())
            elif p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(p)
                        target = self.data_dir / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if str(f.resolve()) != str(target.resolve()):
                            target.write_bytes(f.read_bytes())

        return _ingest(
            data_path=self.data_dir,
            clean=True,
            reindex=reindex,
            embedding_model=self.embedding_model,
            embedder=self.embedder,
        )

    def query(self, question: str, *, multi_query: bool = True) -> RAGAnswer:
        """Run a full RAG query and return a friendly answer object."""
        resp: RAGResponse = _query(
            question,
            multi_query=multi_query,
            use_guardrails=True,
            use_retry=True,
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            llm=self.llm,
            embedder=self.embedder,
        )
        return RAGAnswer(
            text=resp.answer,
            sources=resp.sources,
            evaluation=resp.evaluation,
            retried=resp.retried,
        )

