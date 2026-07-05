"""User-facing documentation printed by ``rag-python docs``."""
from __future__ import annotations

DOCS_BASE_URL = "https://github.com/RaghavOG/rag-python/tree/main/docs"

CLI_EPILOG = """
examples:
  rag-python ingest ./data --reindex
  rag-python query "How many days of annual leave?"
  rag-python query "leave policy" --stream -v
  rag-python query "benefits" --retriever hybrid
  rag-python docs quickstart
  rag-python docs --list

online docs:
  {base_url}
""".format(base_url=DOCS_BASE_URL).strip()

TOPICS: dict[str, str] = {
    "quickstart": """
rag-python — Quick start
========================

1. Install
   pip install rag-python

2. Set your API key (OpenAI default)
   export OPENAI_API_KEY=sk-...

3. Ingest documents (TXT, MD, PDF, DOCX, CSV, JSON, HTML)
   rag-python ingest ./my-docs --reindex

4. Ask a question
   rag-python query "What is our leave policy?"
   rag-python query "annual leave" --stream -v

Python API
----------
  from rag_python import RAG

  rag = RAG(llm_model="gpt-4o-mini")
  rag.ingest(["./my-docs"], reindex=True)
  print(rag.query("What is our leave policy?").text)

More: rag-python docs install | cli | config | providers | features
Online: {base_url}
""".format(base_url=DOCS_BASE_URL).strip(),
    "install": """
rag-python — Install & optional extras
======================================

Base install (OpenAI + Chroma + document loaders):
  pip install rag-python

Optional extras:
  pip install rag-python[local]      Offline embeddings (sentence-transformers)
  pip install rag-python[hybrid]     BM25 + vector hybrid search
  pip install rag-python[rerank]     Cross-encoder reranking
  pip install rag-python[anthropic]  Claude LLM
  pip install rag-python[gemini]     Gemini LLM
  pip install rag-python[all]        All optional features

From source:
  git clone https://github.com/RaghavOG/rag-python.git
  cd rag-python
  pip install -e ".[dev,all]"

Verify:
  rag-python --version
  pytest
""".strip(),
    "cli": """
rag-python — CLI reference
==========================

USAGE
  rag-python ingest PATH [PATH ...] [options]
  rag-python query QUESTION [options]
  rag-python docs [TOPIC]
  rag-python --version

INGEST
  paths                 Files or directories to ingest
  --reindex             Clear vector store before ingesting
  --llm-provider        openai | azure_openai | anthropic | gemini | ollama
  --embedding-provider  openai | azure_openai | ollama | local
  --llm-model           Model or deployment name
  --embedding-model     Embedding model name
  --openai-api-key      Override OPENAI_API_KEY
  --ollama-base-url     Ollama URL (default http://localhost:11434)

QUERY
  question              Natural-language question (words joined if multiple)
  --retriever           vector | multi_query | hybrid (default: multi_query)
  --no-multi-query      Shortcut for --retriever vector
  --metadata-filter     JSON Chroma filter, e.g. '{"filename": "policy.pdf"}'
  --stream              Stream answer tokens to stdout
  -v, --verbose         Show evaluation scores and source paths

ENVIRONMENT
  OPENAI_API_KEY        Required for default OpenAI provider
  RAG_PYTHON_DATA_DIR   Document storage (default ./data)
  RAG_PYTHON_CHROMA_DIR Vector DB path (default ./chroma_db)

Examples:
  rag-python ingest ./policies ./handbook.pdf --reindex
  rag-python query "How many vacation days?" -v
  rag-python query "PTO policy" --retriever hybrid --stream
  rag-python query "benefits" --metadata-filter '{"filename": "hr.pdf"}'
""".strip(),
    "config": """
rag-python — Configuration
==========================

Environment variables (.env supported via python-dotenv):

  OPENAI_API_KEY          OpenAI LLM + embeddings
  ANTHROPIC_API_KEY       Claude (LLM only)
  GEMINI_API_KEY          Gemini (LLM only)
  AZURE_OPENAI_ENDPOINT   Azure OpenAI
  AZURE_OPENAI_API_KEY    Azure OpenAI
  OPENAI_API_VERSION      Azure API version
  OLLAMA_BASE_URL         Local Ollama server
  LOCAL_EMBEDDING_MODEL   Model for embedding_provider=local
  RAG_PYTHON_DATA_DIR     Default ./data
  RAG_PYTHON_CHROMA_DIR   Default ./chroma_db

  CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_STRATEGY
  TOP_K_RETRIEVE, TOP_K_RERANK, MULTI_QUERY_N
  GUARDRAILS_ENABLED, MAX_RETRIES, RERANKER_MODEL

Python RAGConfig:
  from rag_python import RAG, RAGConfig, ChunkingConfig, SearchConfig

  rag = RAG(
      config=RAGConfig(
          chunking=ChunkingConfig(strategy="recursive", chunk_size=512),
          search=SearchConfig(
              retriever="hybrid",
              top_k_retrieve=20,
              metadata_filter={"filename": "policy.pdf"},
          ),
      ),
  )

Shorthand on RAG():
  chunk_strategy, chunk_size, retriever, metadata_filter,
  top_k_retrieve, document_extensions, ...

Logging:
  import rag_python
  rag_python.configure_logging()
""".strip(),
    "providers": """
rag-python — Providers
======================

LLM (generation, rewriting, guardrails):
  openai         OPENAI_API_KEY (default)
  azure_openai   AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY
  anthropic      ANTHROPIC_API_KEY + pip install rag-python[anthropic]
  gemini         GEMINI_API_KEY + pip install rag-python[gemini]
  ollama         Local Ollama — set --llm-model to your model name

Embeddings (retrieval):
  openai         OPENAI_API_KEY (default)
  azure_openai   Azure deployment for embeddings
  ollama         Local embedding model via Ollama
  local          Offline sentence-transformers + pip install rag-python[local]

Common combos:
  OpenAI end-to-end:
    RAG(llm_provider="openai", embedding_provider="openai")

  Claude + OpenAI embeddings:
    RAG(llm_provider="anthropic", llm_model="claude-opus-4-6",
        embedding_provider="openai")

  Fully local embeddings:
    RAG(llm_provider="ollama", llm_model="llama3.1",
        embedding_provider="local", embedding_model="all-MiniLM-L6-v2")
""".strip(),
    "features": """
rag-python — Features
=====================

Ingest pipeline
  Loaders: .txt .md .pdf .docx .csv .json .html
  Cleaning, chunking (recursive | structure_aware | semantic)
  Embeddings → ChromaDB vector store

Query pipeline
  Query rewriting + multi-query retrieval
  Hybrid search: BM25 + vector (pip install rag-python[hybrid])
  Cross-encoder reranking (pip install rag-python[rerank])
  Metadata filters on retrieval (source, filename, ...)
  Streaming answers: rag.query_stream()
  Guardrails: prompt injection + hallucination checks
  Evaluation + self-correction retry loop

CLI
  rag-python ingest | query | docs
  rag-python --version

Docs
  rag-python docs [topic]
  https://github.com/RaghavOG/rag-python/tree/main/docs
""".strip(),
}


def list_topics() -> list[str]:
    return sorted(TOPICS.keys())


def print_topic(topic: str) -> None:
    key = topic.lower().strip()
    if key not in TOPICS:
        available = ", ".join(list_topics())
        raise SystemExit(f"Unknown topic '{topic}'. Available: {available}\nUse: rag-python docs --list")
    print(TOPICS[key])


def print_topic_list() -> None:
    print("rag-python documentation topics:\n")
    for name in list_topics():
        print(f"  {name}")
    print("\nUsage: rag-python docs <topic>")
    print(f"Online: {DOCS_BASE_URL}")
