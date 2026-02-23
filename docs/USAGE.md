## Usage

### Python API (recommended)

```python
from complete_basic_rag import RAG

rag = RAG(
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)

rag.ingest(["./data"], reindex=True)
ans = rag.query("How many days of annual leave?")
print(ans.text)
```

### CLI

Install editable first:

```bash
pip install -e .
```

Then:

```bash
complete-basic-rag ingest ./data --reindex
complete-basic-rag query "How many days of annual leave?"
```

