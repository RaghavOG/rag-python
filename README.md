## complete-basic-rag

A **modular Retrieval-Augmented Generation (RAG)** backend using **OpenAI** for LLM + embeddings and **ChromaDB** for vector search, designed to be extended later with a separate frontend (web UI, dashboard, etc.).

### Backend (current)

- **Tech**: Python, OpenAI API, ChromaDB, sentence-transformers reranker.
- **Pipeline**:
  - **Documents**: `document_loaders.py` → `cleaning.py` → `chunking.py` → `embeddings.py` → `vector_store.py`
  - **Query**: `query_rewriting.py` → `retrieval.py` (+ `reranker.py`)
  - **Generation**: `generation.py` with **guardrails** in `guardrails.py`
  - **Eval / Retry**: `evaluation.py` + orchestration in `rag_pipeline.py`
  - **CLI entrypoint**: `main.py`

#### Setup

1. **Install dependencies** (recommended: inside a virtualenv):

```bash
cd complete-basic-rag
pip install -r requirements.txt
```

2. **Configure OpenAI**:

- Copy `.env.example` → `.env`
- Fill in `OPENAI_API_KEY`.

3. **Ingest documents** (a sample doc is in `data/sample.txt`):

```bash
python main.py ingest --reindex --strategy recursive
```

4. **Ask questions**:

```bash
python main.py query "How many days of annual leave?"
python main.py chat          # interactive mode
python main.py chat -v       # with scores & sources
```

### Frontend (later)

The repo is structured to support a **separate frontend** (e.g. React/Vue/Svelte) alongside the backend:

- `frontend/` – reserved for a future SPA / dashboard.
- `.gitignore` already includes **Node / build artifacts** (`node_modules/`, `.vite/`, `.next/`, etc.).

When you add a frontend, you can:

- Put its source under `frontend/` (with its own `package.json`, build tooling, etc.).
- Optionally expose the backend via a REST/GraphQL API or websocket layer on top of the existing RAG pipeline.

### Repo structure

```text
.
├─ main.py                 # CLI entrypoint (backend)
├─ config.py               # Settings / env
├─ rag_pipeline.py         # Orchestrates full RAG flow
├─ document_loaders.py     # Document loaders
├─ cleaning.py             # Cleaning / normalization
├─ chunking.py             # Chunking strategies
├─ embeddings.py           # OpenAI embeddings
├─ vector_store.py         # ChromaDB vector store
├─ query_rewriting.py      # Query understanding & multi-query
├─ retrieval.py            # Retrieval + reranking glue
├─ reranker.py             # Cross-encoder reranker
├─ generation.py           # LLM generation
├─ guardrails.py           # Input/output guardrails
├─ evaluation.py           # RAG evaluation & retry
├─ data/
│  └─ sample.txt           # Example document
├─ frontend/               # (empty for now) future UI
├─ requirements.txt
├─ .env.example
└─ .gitignore
```

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

