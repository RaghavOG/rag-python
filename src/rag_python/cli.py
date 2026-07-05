"""rag-python command-line interface."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from . import __version__
from .client import RAG
from .help_text import CLI_EPILOG, list_topics, print_topic, print_topic_list


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
        metavar="PROVIDER",
        help="LLM backend (default: openai). See: rag-python docs providers",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        metavar="MODEL",
        help="LLM model or Azure deployment name (default: from env LLM_MODEL)",
    )
    parser.add_argument(
        "--embedding-provider",
        default="openai",
        choices=["openai", "azure_openai", "ollama", "local"],
        metavar="PROVIDER",
        help="Embedding backend (default: openai). Use local for offline embeddings",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        metavar="MODEL",
        help="Embedding model name (default: from env EMBEDDING_MODEL)",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        metavar="URL",
        help="Ollama server URL (default: http://localhost:11434 or OLLAMA_BASE_URL)",
    )
    parser.add_argument("--azure-endpoint", default=None, help="Azure OpenAI endpoint URL")
    parser.add_argument("--azure-api-key", default=None, help="Azure OpenAI API key")
    parser.add_argument(
        "--azure-api-version",
        default=None,
        help="Azure API version (default: 2023-09-01-preview)",
    )
    parser.add_argument("--openai-api-key", default=None, help="OpenAI API key (overrides env)")
    parser.add_argument("--anthropic-api-key", default=None, help="Anthropic API key")
    parser.add_argument("--gemini-api-key", default=None, help="Gemini API key")


def _add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retriever",
        choices=["vector", "multi_query", "hybrid"],
        default=None,
        metavar="MODE",
        help=(
            "Retrieval mode: vector (single query), multi_query (default, with rewriting), "
            "or hybrid (BM25+vector; requires pip install rag-python[hybrid])"
        ),
    )
    parser.add_argument(
        "--metadata-filter",
        type=_parse_metadata_filter,
        default=None,
        metavar="JSON",
        help='Filter chunks by metadata, e.g. \'{"filename": "policy.pdf"}\'',
    )


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-python",
        description=(
            "Production-grade RAG for Python — ingest documents, ask questions, "
            "get grounded answers with multi-LLM support."
        ),
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    ing = sub.add_parser(
        "ingest",
        help="Load files into the vector store (chunk + embed)",
        description=(
            "Ingest one or more files or directories into the ChromaDB vector store.\n"
            "Supported formats: .txt .md .pdf .docx .csv .json .html"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  rag-python ingest ./data --reindex\n"
            "  rag-python ingest policy.pdf handbook/ --embedding-provider local"
        ),
    )
    ing.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="File or directory paths to ingest",
    )
    ing.add_argument(
        "--reindex",
        action="store_true",
        help="Delete existing vectors before ingesting (fresh index)",
    )
    _add_provider_args(ing)

    q = sub.add_parser(
        "query",
        help="Ask a question against ingested documents",
        description=(
            "Run the full RAG pipeline: retrieve relevant chunks, generate an answer, "
            "optionally stream tokens and show sources."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  rag-python query "How many days of annual leave?"\n'
            "  rag-python query \"PTO policy\" --stream -v\n"
            '  rag-python query "benefits" --retriever hybrid --metadata-filter \'{"filename": "hr.pdf"}\''
        ),
    )
    q.add_argument(
        "question",
        nargs="+",
        metavar="QUESTION",
        help="Question text (multiple words are joined)",
    )
    q.add_argument(
        "--no-multi-query",
        action="store_true",
        help="Use single-query vector retrieval (same as --retriever vector)",
    )
    q.add_argument(
        "--stream",
        action="store_true",
        help="Stream answer tokens to stdout as they are generated",
    )
    q.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="After the answer, print evaluation scores and top source paths",
    )
    _add_provider_args(q)
    _add_search_args(q)

    docs = sub.add_parser(
        "docs",
        help="Show user documentation in the terminal",
        description="Print built-in help topics. Full docs: https://github.com/RaghavOG/rag-python/tree/main/docs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="topics: " + ", ".join(list_topics()),
    )
    docs.add_argument(
        "topic",
        nargs="?",
        default="quickstart",
        choices=list_topics(),
        metavar="TOPIC",
        help="Documentation topic (default: quickstart)",
    )
    docs.add_argument(
        "--list",
        action="store_true",
        help="List all available documentation topics",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _make_parser()
    args = parser.parse_args(argv)

    if args.command == "docs":
        if args.list:
            print_topic_list()
        else:
            print_topic(args.topic)
        return

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
        if args.stream:
            stream = rag.query_stream(question, search=search)
            for token in stream:
                print(token, end="", flush=True)
            print()
            result = stream.result
            if args.verbose:
                print("\n--- evaluation ---")
                print(result.evaluation)
                print("\n--- sources ---")
                for s in result.sources[:5]:
                    print(s.get("metadata", {}).get("source", ""), "score:", s.get("score"))
            return

        ans = rag.query(question, search=search)
        print(ans.text)
        if args.verbose:
            print("\n--- evaluation ---")
            print(ans.evaluation)
            print("\n--- sources ---")
            for s in ans.sources[:5]:
                print(s.get("metadata", {}).get("source", ""), "score:", s.get("score"))


if __name__ == "__main__":
    main(sys.argv[1:])
