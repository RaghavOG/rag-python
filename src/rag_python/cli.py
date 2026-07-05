"""rag-python command-line interface."""
import argparse
import json
from dataclasses import replace

from . import __version__
from .client import RAG


def _build_rag(args: argparse.Namespace) -> RAG:
    kwargs: dict = dict(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        openai_api_key=args.openai_api_key,
        azure_endpoint=args.azure_endpoint,
        azure_api_key=args.azure_api_key,
        azure_api_version=args.azure_api_version,
        anthropic_api_key=args.anthropic_api_key,
        gemini_api_key=args.gemini_api_key,
        ollama_base_url=args.ollama_base_url,
    )
    if getattr(args, "retriever", None):
        kwargs["retriever"] = args.retriever
    if getattr(args, "metadata_filter", None):
        kwargs["metadata_filter"] = args.metadata_filter
    return RAG(**kwargs)


def _parse_metadata_filter(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON for metadata filter: {e}") from e


def _add_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-provider",
        default="openai",
        choices=["openai", "azure_openai", "anthropic", "gemini", "ollama"],
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--embedding-provider",
        default="openai",
        choices=["openai", "azure_openai", "ollama", "local"],
    )
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--ollama-base-url", default=None)
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--azure-api-key", default=None)
    parser.add_argument("--azure-api-version", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--anthropic-api-key", default=None)
    parser.add_argument("--gemini-api-key", default=None)


def _add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retriever",
        choices=["vector", "multi_query", "hybrid"],
        default=None,
        help="Retrieval strategy (default: multi_query; hybrid needs pip install rag-python[hybrid])",
    )
    parser.add_argument(
        "--metadata-filter",
        type=_parse_metadata_filter,
        default=None,
        help='Chroma metadata filter as JSON, e.g. \'{"filename": "policy.pdf"}\'',
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag-python",
        description="rag-python — modular RAG with query rewriting, reranking, guardrails, and multi-LLM support.",
    )
    parser.add_argument("--version", action="version", version=f"rag-python {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="Ingest files/folders into the vector store")
    ing.add_argument("paths", nargs="+", help="Files or folders to ingest")
    ing.add_argument("--reindex", action="store_true", help="Clear vector store and re-ingest")
    _add_provider_args(ing)

    q = sub.add_parser("query", help="Ask a question against ingested documents")
    q.add_argument("question", nargs="+", help="Question text")
    q.add_argument("--no-multi-query", action="store_true", help="Use vector retriever only")
    q.add_argument("-v", "--verbose", action="store_true")
    _add_provider_args(q)
    _add_search_args(q)

    args = parser.parse_args()

    if args.command == "ingest":
        rag = _build_rag(args)
        n = rag.ingest(args.paths, reindex=args.reindex)
        print(f"Ingested {n} chunks.")
        return

    if args.command == "query":
        rag = _build_rag(args)
        question = " ".join(args.question)
        retriever = args.retriever
        if retriever is None and args.no_multi_query:
            retriever = "vector"
        search = replace(
            rag.config.search,
            retriever=retriever or rag.config.search.retriever,
            metadata_filter=args.metadata_filter or rag.config.search.metadata_filter,
        )
        ans = rag.query(question, search=search)
        print(ans.text)
        if args.verbose:
            print("\n--- evaluation ---")
            print(ans.evaluation)
            print("\n--- sources ---")
            for s in ans.sources[:5]:
                print(s.get("metadata", {}).get("source", ""), "score:", s.get("score"))


if __name__ == "__main__":
    main()
