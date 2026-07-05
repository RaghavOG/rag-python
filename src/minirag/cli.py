"""MiniRAG command-line interface."""
import argparse

from .client import RAG


def _build_rag(args: argparse.Namespace) -> RAG:
    return RAG(
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


def _add_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-provider", default="openai", choices=["openai", "azure_openai", "anthropic", "gemini", "ollama"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--embedding-provider", default="openai", choices=["openai", "azure_openai", "ollama"])
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--ollama-base-url", default=None)
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--azure-api-key", default=None)
    parser.add_argument("--azure-api-version", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--anthropic-api-key", default=None)
    parser.add_argument("--gemini-api-key", default=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="minirag",
        description="MiniRAG — modular RAG with query rewriting, reranking, guardrails, and multi-LLM support.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="Ingest files/folders into the vector store")
    ing.add_argument("paths", nargs="+", help="Files or folders to ingest")
    ing.add_argument("--reindex", action="store_true", help="Clear vector store and re-ingest")
    _add_provider_args(ing)

    q = sub.add_parser("query", help="Ask a question against ingested documents")
    q.add_argument("question", nargs="+", help="Question text")
    q.add_argument("--no-multi-query", action="store_true")
    q.add_argument("-v", "--verbose", action="store_true")
    _add_provider_args(q)

    args = parser.parse_args()

    if args.command == "ingest":
        rag = _build_rag(args)
        n = rag.ingest(args.paths, reindex=args.reindex)
        print(f"Ingested {n} chunks.")
        return

    if args.command == "query":
        rag = _build_rag(args)
        question = " ".join(args.question)
        ans = rag.query(question, multi_query=not args.no_multi_query)
        print(ans.text)
        if args.verbose:
            print("\n--- evaluation ---")
            print(ans.evaluation)
            print("\n--- sources ---")
            for s in ans.sources[:5]:
                print(s.get("metadata", {}).get("source", ""), "score:", s.get("score"))


if __name__ == "__main__":
    main()
