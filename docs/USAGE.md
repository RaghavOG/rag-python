## Usage

### Python API

```python
from rag_python import RAG

rag = RAG(
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)

rag.ingest(["./data"], reindex=True)
ans = rag.query("How many days of annual leave?")
print(ans.text)
print(ans.sources)      # retrieved chunks
print(ans.evaluation)   # faithfulness / relevance scores
```

### Hybrid retrieval

Install optional BM25 support:

```bash
pip install rag-python[hybrid]
```

```python
from rag_python import RAG, SearchConfig

rag = RAG(retriever="hybrid")
rag.ingest(["./data"], reindex=True)

# Optional: restrict retrieval to chunks from a specific file
ans = rag.query(
    "annual leave",
    search=SearchConfig(retriever="hybrid", metadata_filter={"filename": "hr-policy.txt"}),
)
```

### CLI

```bash
pip install -e .
rag-python ingest ./data --reindex
rag-python query "How many days of annual leave?" -v
rag-python query "annual leave" --retriever hybrid
```

### Local development

```bash
pip install -e ".[dev,rerank,hybrid]"
pytest
python main.py ingest --reindex
python main.py chat
```
