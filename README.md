## complete-basic-rag

A **packaged, modular RAG toolkit** with:

- **Query understanding / rewriting**
- **Multi-query retrieval**
- **Reranking**
- **LLM generation**
- **Guardrails** (prompt injection + groundedness)
- **Evaluation + retry/self-correction**
- **Pluggable LLM providers**: OpenAI, Azure OpenAI, Anthropic, Gemini, Ollama

### Install

For local dev:

```bash
pip install -r requirements.txt
```

Or install as a package (editable):

```bash
pip install -e .
```

Optional provider extras:

```bash
pip install -e ".[anthropic]"
pip install -e ".[gemini]"
```

### Quickstart (Python API)

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

### Quickstart (CLI)

After `pip install -e .` you get a console script:

```bash
complete-basic-rag ingest ./data --reindex
complete-basic-rag query "How many days of annual leave?"
```

### Providers

You can mix-and-match **LLM provider** and **embedding provider**.

- **OpenAI (default)**:
  - Env: `OPENAI_API_KEY`
  - Example:

```python
rag = RAG(llm_provider="openai", llm_model="gpt-4o-mini", embedding_provider="openai", embedding_model="text-embedding-3-small")
```

- **Azure OpenAI** (model = deployment name):
  - Env: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`
  - Example:

```python
rag = RAG(
  llm_provider="azure_openai",
  llm_model="my-gpt-deployment",
  embedding_provider="azure_openai",
  embedding_model="my-embed-deployment",
)
```

- **Anthropic (Claude)** (embeddings still need OpenAI/Azure/Ollama):

```python
rag = RAG(
  llm_provider="anthropic",
  llm_model="claude-opus-4-6",
  embedding_provider="openai",
  embedding_model="text-embedding-3-small",
)
```

- **Gemini** (embeddings still need OpenAI/Azure/Ollama):

```python
rag = RAG(
  llm_provider="gemini",
  llm_model="gemini-2.0-flash",
  embedding_provider="openai",
  embedding_model="text-embedding-3-small",
)
```

- **Ollama (local)**:
  - Needs Ollama running (default `http://localhost:11434`)

```python
rag = RAG(
  llm_provider="ollama",
  llm_model="llama3.1",
  embedding_provider="ollama",
  embedding_model="mxbai-embed-large",
)
```

### Frontend (later)

The repo is structured to support a **separate frontend** (e.g. React/Vue/Svelte) alongside the backend:

- `frontend/` – reserved for a future SPA / dashboard.
- `.gitignore` already includes **Node / build artifacts** (`node_modules/`, `.vite/`, `.next/`, etc.).

When you add a frontend, you can:

- Put its source under `frontend/` (with its own `package.json`, build tooling, etc.).
- Optionally expose the backend via a REST/GraphQL API or websocket layer on top of the existing RAG pipeline.

### Repo structure (high level)

```text
.
├─ main.py                 # CLI entrypoint (backend)
├─ config.py               # Settings / env
├─ complete_basic_rag/      # Public Python package API + CLI
├─ backend/                 # Implementation (RAG pipeline + providers)
│  └─ providers/            # OpenAI / Azure / Anthropic / Gemini / Ollama
├─ data/
│  └─ sample.txt           # Example document
├─ frontend/               # (empty for now) future UI
├─ requirements.txt
├─ pyproject.toml           # Packaging config
├─ .env.example
└─ .gitignore
```

### Docs

- `docs/USAGE.md`
- `docs/PROVIDERS.md`

### Git & GitHub

This folder is already initialized as a git repository (`git init` run from the root).  
Next steps for GitHub:

- Create a new GitHub repo.
- Add this directory as the remote:

```bash
git add .
git commit -m "Initial RAG backend"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

