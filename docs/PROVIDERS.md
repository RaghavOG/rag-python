## Providers

This package separates:

- **LLM provider** (generation, query rewriting, guardrails, evaluation)
- **Embedding provider** (document + query embeddings for retrieval)

### Supported LLM providers

- `openai`
- `azure_openai` (model = deployment name)
- `anthropic`
- `gemini`
- `ollama` (local)

### Supported embedding providers

- `openai`
- `azure_openai` (model = embedding deployment name)
- `ollama` (local embedding model)

### Environment variables

- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Gemini**: `GEMINI_API_KEY`
- **Azure OpenAI**: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`
- **Ollama**: `OLLAMA_BASE_URL` (default `http://localhost:11434`)

### Notes / gotchas

- **Anthropic/Gemini**: this repo currently uses them for LLM tasks only; embeddings should come from `openai`, `azure_openai`, or `ollama`.
- **Ollama**: you must have the model pulled locally (e.g. `ollama pull llama3.1`) and an embedding model for `embed` (e.g. `ollama pull mxbai-embed-large`).

