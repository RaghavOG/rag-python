import argparse
from pathlib import Path

from . import RAG


def main() -> None:
    parser = argparse.ArgumentParser(description="complete-basic-rag CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="Ingest files/folders into the vector store")
    ing.add_argument("paths", nargs="+", help="Files/folders to ingest")
    ing.add_argument("--reindex", action="store_true", help="Clear and re-ingest")
    ing.add_argument("--llm-provider", default="openai", choices=["openai", "azure_openai", "anthropic", "gemini", "ollama"])
    ing.add_argument("--llm-model", default=None)
    ing.add_argument("--embedding-provider", default="openai", choices=["openai", "azure_openai", "ollama"])
    ing.add_argument("--embedding-model", default=None)
    ing.add_argument("--ollama-base-url", default=None)
    ing.add_argument("--azure-endpoint", default=None)
    ing.add_argument("--azure-api-key", default=None)
    ing.add_argument("--azure-api-version", default=None)
    ing.add_argument("--openai-api-key", default=None)
    ing.add_argument("--anthropic-api-key", default=None)
    ing.add_argument("--gemini-api-key", default=None)

    q = sub.add_parser("query", help="Ask a question")
    q.add_argument("question", nargs="+", help="Question text")
    q.add_argument("--llm-provider", default="openai", choices=["openai", "azure_openai", "anthropic", "gemini", "ollama"])
    q.add_argument("--llm-model", default=None)
    q.add_argument("--embedding-provider", default="openai", choices=["openai", "azure_openai", "ollama"])
    q.add_argument("--embedding-model", default=None)
    q.add_argument("--ollama-base-url", default=None)
    q.add_argument("--azure-endpoint", default=None)
    q.add_argument("--azure-api-key", default=None)
    q.add_argument("--azure-api-version", default=None)
    q.add_argument("--openai-api-key", default=None)
    q.add_argument("--anthropic-api-key", default=None)
    q.add_argument("--gemini-api-key", default=None)
    q.add_argument("--no-multi-query", action="store_true")
    q.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.command == "ingest":
        rag = RAG(
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
        n = rag.ingest(args.paths, reindex=args.reindex)
        print(f"Ingested {n} chunks.")
        return

    if args.command == "query":
        rag = RAG(
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
        question = " ".join(args.question)
        ans = rag.query(question, multi_query=not args.no_multi_query)
        print(ans.text)
        if args.verbose:
            print("\n--- evaluation ---")
            print(ans.evaluation)
        return


if __name__ == "__main__":
    main()

