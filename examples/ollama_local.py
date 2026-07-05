from rag_python import RAG

rag = RAG(
    llm_provider="ollama",
    llm_model="llama3.1",
    embedding_provider="ollama",
    embedding_model="mxbai-embed-large",
)

rag.ingest(["./data"], reindex=True)
answer = rag.query("How many days of annual leave?")
print(answer.text)
