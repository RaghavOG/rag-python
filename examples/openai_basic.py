from complete_basic_rag import RAG


def main():
    rag = RAG(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
    )
    rag.ingest(["./data"], reindex=True)
    ans = rag.query("How many days of annual leave?")
    print(ans.text)


if __name__ == "__main__":
    main()

