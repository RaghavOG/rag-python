"""Vector store (Chroma) for ingestion and retrieval."""
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from config import CHROMA_PERSIST_DIR
from .embeddings import embed_texts

COLLECTION_NAME = "rag_docs"
_chroma_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        p = Path(CHROMA_PERSIST_DIR)
        p.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(p), settings=Settings(anonymized_telemetry=False))
    return _chroma_client


def get_collection():
    return _get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_chunks(
    ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, Any]],
    embed_model: str | None = None,
) -> None:
    """Embed chunks and add to Chroma."""
    embeddings = embed_texts(texts, model=embed_model)
    coll = get_collection()
    clean_metas = []
    for m in metadatas:
        clean = {}
        for k, v in m.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        clean_metas.append(clean)
    coll.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=clean_metas)


def retrieve(
    query_embedding: list[float],
    top_k: int = 20,
    where: dict | None = None,
) -> list[tuple[str, dict, float]]:
    """Return (document, metadata, distance) for top_k. Lower distance = more similar."""
    coll = get_collection()
    res = coll.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0] or []
    metas = res["metadatas"][0] or []
    dists = res["distances"][0] or []
    return list(zip(docs, metas, dists))


def delete_all() -> None:
    """Remove all documents from the collection (for re-ingestion)."""
    _get_client().delete_collection(COLLECTION_NAME)

