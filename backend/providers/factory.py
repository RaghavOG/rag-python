from __future__ import annotations

import os
from typing import Literal

from .base import LLMProvider, EmbeddingProvider
from .openai_provider import OpenAIProvider
from .azure_openai_provider import AzureOpenAIProvider
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .ollama_provider import OllamaProvider


LLMProviderName = Literal["openai", "azure_openai", "anthropic", "gemini", "ollama"]
EmbeddingProviderName = Literal["openai", "azure_openai", "ollama"]


def make_llm_provider(name: LLMProviderName, **kwargs) -> LLMProvider:
    if name == "openai":
        return OpenAIProvider(api_key=kwargs.get("api_key"))
    if name == "azure_openai":
        return AzureOpenAIProvider(
            azure_endpoint=kwargs.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=kwargs.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_version=kwargs.get("api_version") or os.getenv("OPENAI_API_VERSION", "2023-09-01-preview"),
        )
    if name == "anthropic":
        return AnthropicProvider(api_key=kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY"))
    if name == "gemini":
        return GeminiProvider(
            api_key=kwargs.get("api_key") or os.getenv("GEMINI_API_KEY"),
            vertexai=bool(kwargs.get("vertexai", False)),
            project=kwargs.get("project") or os.getenv("VERTEX_PROJECT"),
            location=kwargs.get("location") or os.getenv("VERTEX_LOCATION"),
        )
    if name == "ollama":
        return OllamaProvider(base_url=kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    raise ValueError(f"Unknown LLM provider: {name}")


def make_embedding_provider(name: EmbeddingProviderName, **kwargs) -> EmbeddingProvider:
    if name == "openai":
        return OpenAIProvider(api_key=kwargs.get("api_key"))
    if name == "azure_openai":
        return AzureOpenAIProvider(
            azure_endpoint=kwargs.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=kwargs.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_version=kwargs.get("api_version") or os.getenv("OPENAI_API_VERSION", "2023-09-01-preview"),
        )
    if name == "ollama":
        return OllamaProvider(base_url=kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    raise ValueError(f"Unknown embedding provider: {name}")

