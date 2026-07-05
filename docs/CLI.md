# CLI reference

The `rag-python` command is installed with the package:

```bash
pip install rag-python
rag-python --help
```

## Commands

| Command | Description |
|---------|-------------|
| `rag-python ingest` | Load documents into the vector store |
| `rag-python query` | Ask a question against ingested documents |
| `rag-python docs` | Show built-in documentation in the terminal |
| `rag-python --version` | Print package version |

---

## `rag-python ingest`

Load files or folders, chunk them, embed them, and store vectors in ChromaDB.

```bash
rag-python ingest PATH [PATH ...] [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `PATH` | One or more files or directories |

### Options

| Option | Description |
|--------|-------------|
| `--reindex` | Clear the vector store before ingesting |
| `--llm-provider` | `openai` (default), `azure_openai`, `anthropic`, `gemini`, `ollama` |
| `--embedding-provider` | `openai` (default), `azure_openai`, `ollama`, `local` |
| `--llm-model` | Model or Azure deployment name |
| `--embedding-model` | Embedding model name |
| `--openai-api-key` | Override `OPENAI_API_KEY` |
| `--ollama-base-url` | Ollama server URL |

### Supported file types

`.txt` `.md` `.pdf` `.docx` `.csv` `.json` `.html`

### Examples

```bash
rag-python ingest ./data --reindex
rag-python ingest policy.pdf ./handbook --embedding-provider local
```

---

## `rag-python query`

Run retrieval-augmented generation against ingested documents.

```bash
rag-python query QUESTION [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `QUESTION` | Natural-language question (multiple words are joined) |

### Options

| Option | Description |
|--------|-------------|
| `--retriever` | `vector`, `multi_query` (default), or `hybrid` |
| `--no-multi-query` | Shortcut for `--retriever vector` |
| `--metadata-filter` | JSON Chroma metadata filter |
| `--stream` | Stream answer tokens to stdout |
| `-v`, `--verbose` | Print evaluation scores and source paths |
| `--llm-provider` | LLM backend (see ingest) |
| `--embedding-provider` | Embedding backend (see ingest) |

### Examples

```bash
rag-python query "How many days of annual leave?"
rag-python query "PTO policy" --stream -v
rag-python query "benefits" --retriever hybrid
rag-python query "salary" --metadata-filter '{"filename": "hr-policy.pdf"}'
rag-python query "remote work" --llm-provider ollama --llm-model llama3.1
```

### Metadata filter

Restrict retrieval to chunks matching Chroma metadata:

```bash
rag-python query "leave" --metadata-filter '{"filename": "policy.pdf"}'
rag-python query "docs" --metadata-filter '{"source": "/path/to/file.txt"}'
```

---

## `rag-python docs`

Print user documentation in the terminal (no browser required).

```bash
rag-python docs [TOPIC]
rag-python docs --list
```

### Topics

| Topic | Content |
|-------|---------|
| `quickstart` | Install, ingest, query (default) |
| `install` | pip extras and source install |
| `cli` | Full CLI reference |
| `config` | Environment variables and `RAGConfig` |
| `providers` | LLM and embedding providers |
| `features` | Pipeline overview |

### Examples

```bash
rag-python docs
rag-python docs cli
rag-python docs --list
```

---

## Environment variables

| Variable | Used for |
|----------|----------|
| `OPENAI_API_KEY` | Default OpenAI LLM + embeddings |
| `ANTHROPIC_API_KEY` | Claude |
| `GEMINI_API_KEY` | Gemini |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI |
| `OLLAMA_BASE_URL` | Local Ollama (default `http://localhost:11434`) |
| `RAG_PYTHON_DATA_DIR` | Default document directory (`./data`) |
| `RAG_PYTHON_CHROMA_DIR` | Vector store path (`./chroma_db`) |

See [Configuration](CONFIGURATION.md) for the full list.
