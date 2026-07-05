"""Thin CLI wrapper for local development (prefer: ragkit CLI after pip install -e .)."""
import argparse
from pathlib import Path

from ragkit.config import DATA_DIR, CHUNK_STRATEGY
from ragkit import ingest, query, RAGResponse


def cmd_ingest(args):
    path = Path(args.path) if args.path else DATA_DIR
    path.mkdir(parents=True, exist_ok=True)
    n = ingest(
        data_path=path,
        clean=True,
        chunk_strategy=args.strategy or CHUNK_STRATEGY,
        reindex=args.reindex,
    )
    print(f"Ingested {n} chunks from {path}")


def cmd_query(args):
    resp: RAGResponse = query(
        args.question,
        multi_query=not args.no_multi_query,
        use_guardrails=not args.no_guardrails,
        use_retry=not args.no_retry,
    )
    print(resp.answer)
    if args.verbose:
        print("\n--- Sources ---")
        for s in resp.sources[:5]:
            print(s.get("metadata", {}).get("source", ""), "score:", s.get("score"))
        print("--- Evaluation ---", resp.evaluation, "retried:", resp.retried)


def cmd_chat(args):
    print("RAGKit chat. Type your question. 'quit' or 'exit' to stop.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("quit", "exit", "q"):
            break
        resp = query(q, multi_query=True, use_guardrails=True, use_retry=True)
        print("RAG:", resp.answer)
        if args.verbose:
            print("  [scores]", resp.evaluation)


def main():
    parser = argparse.ArgumentParser(description="RAGKit (local dev CLI)")
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="Load documents, clean, chunk, embed, store")
    ing.add_argument("--path", type=str, default=None, help=f"Data directory (default: {DATA_DIR})")
    ing.add_argument("--reindex", action="store_true", help="Clear and re-ingest")
    ing.add_argument("--strategy", choices=("recursive", "structure_aware", "semantic"), default=None)
    ing.set_defaults(func=cmd_ingest)

    q = sub.add_parser("query", help="Ask a question")
    q.add_argument("question", type=str, nargs="+", help="Question (words joined)")
    q.add_argument("-v", "--verbose", action="store_true")
    q.add_argument("--no-multi-query", action="store_true")
    q.add_argument("--no-guardrails", action="store_true")
    q.add_argument("--no-retry", action="store_true")
    q.set_defaults(func=cmd_query)

    chat = sub.add_parser("chat", help="Interactive Q&A")
    chat.add_argument("-v", "--verbose", action="store_true")
    chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    if args.command == "query":
        args.question = " ".join(args.question)
    args.func(args)


if __name__ == "__main__":
    main()
