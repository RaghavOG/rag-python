from rag_python import RAG

rag = RAG(
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)

rag.ingest(["./data"], reindex=True)
answer = rag.query("How many days of annual leave?")
print(answer.text)
