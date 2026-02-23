from complete_basic_rag import RAG


def main():
    # Ensure Ollama is running and models are pulled:
    #   ollama pull llama3.1
    #   ollama pull mxbai-embed-large
    rag = RAG(
        llm_provider="ollama",
        llm_model="llama3.1",
        embedding_provider="ollama",
        embedding_model="mxbai-embed-large",
    )
    rag.ingest(["./data"], reindex=True)
    ans = rag.query("How many days of annual leave?")
    print(ans.text)


if __name__ == "__main__":
    main()

