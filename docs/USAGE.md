## Usage

### Python API

```python
from minirag import RAG

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

### CLI

```bash
pip install -e .
minirag ingest ./data --reindex
minirag query "How many days of annual leave?" -v
```

### Local development

```bash
pip install -e ".[dev,rerank]"
pytest
python main.py ingest --reindex
python main.py chat
```
