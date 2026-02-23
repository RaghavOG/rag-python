"""Chunking: recursive, structure-aware (headings/sections), and semantic (embedding-based)."""
import re
from dataclasses import dataclass
from typing import Callable

try:
    import tiktoken
except ImportError:
    tiktoken = None


@dataclass
class Chunk:
    """Single chunk with text and metadata."""
    text: str
    metadata: dict


# --- Recursive: split by section → paragraph → sentence → tokens ---
RECURSIVE_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", " ", ""]


def _split_by_tokens(text: str, chunk_size: int, overlap: int, encoding_name: str = "cl100k_base") -> list[str]:
    if not tiktoken:
        size = chunk_size * 4
        overlap_chars = overlap * 4
        out = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            out.append(text[start:end])
            start = end - overlap_chars if end < len(text) else len(text)
        return out
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    out = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        out.append(enc.decode(tokens[start:end]))
        start = end - overlap if end < len(tokens) else len(tokens)
    return out


def _recursive_split(text: str, separators: list[str], chunk_size: int, overlap: int) -> list[str]:
    if not text.strip():
        return []
    sep = separators[0] if separators else ""
    if sep == "":
        return _split_by_tokens(text, chunk_size, overlap)
    parts = text.split(sep)
    if len(parts) == 1:
        return _recursive_split(text, separators[1:], chunk_size, overlap)
    chunks = []
    current = ""
    for p in parts:
        bit = p if sep in "\n" else p + sep
        if len(current) + len(bit) <= chunk_size * 4:
            current += bit
        else:
            if current.strip():
                chunks.append(current.strip())
            current = bit[-overlap * 4 :] + bit if overlap else bit
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_recursive(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Recursive chunking: section → paragraph → sentence → tokens."""
    raw = _recursive_split(text, RECURSIVE_SEPARATORS, chunk_size, overlap)
    meta = dict(metadata or {})
    meta["chunk_strategy"] = "recursive"
    return [Chunk(text=t, metadata={**meta}) for t in raw if t.strip()]


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _structure_sections(text: str) -> list[tuple[str, str]]:
    """Split by markdown-style headings; preserve content under each heading."""
    sections = []
    current_title = "Document"
    current_content = []
    for line in text.splitlines():
        m = HEADING_PATTERN.match(line)
        if m:
            if current_content:
                sections.append((current_title, "\n".join(current_content)))
            current_title = m.group(2).strip()
            current_content = []
        else:
            current_content.append(line)
    if current_content:
        sections.append((current_title, "\n".join(current_content)))
    return sections


def chunk_structure_aware(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Structure-aware: chunk by sections (headings); keep tables/code blocks intact."""
    sections = _structure_sections(text)
    meta = dict(metadata or {})
    meta["chunk_strategy"] = "structure_aware"
    chunks = []
    for title, content in sections:
        content = content.strip()
        if not content:
            continue
        if len(content) <= chunk_size * 4:
            chunks.append(Chunk(text=f"## {title}\n\n{content}", metadata={**meta, "section": title}))
        else:
            sub = _recursive_split(content, RECURSIVE_SEPARATORS[1:], chunk_size, overlap)
            for i, t in enumerate(sub):
                if t.strip():
                    chunks.append(Chunk(
                        text=f"## {title}\n\n{t.strip()}",
                        metadata={**meta, "section": title, "section_part": i},
                    ))
    return chunks


def chunk_semantic(
    text: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: dict | None = None,
    similarity_threshold: float = 0.7,
) -> list[Chunk]:
    """Semantic chunking: approximate topic shifts and split."""
    segments = re.split(r"(?<=[.!?])\s+", text)
    if len(segments) <= 1:
        return chunk_recursive(text, chunk_size, overlap, metadata)

    meta = dict(metadata or {})
    meta["chunk_strategy"] = "semantic"
    chunks = []
    current = []
    current_len = 0
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        current.append(seg)
        current_len += len(seg)
        if current_len >= chunk_size * 3:
            chunk_text = " ".join(current)
            chunks.append(Chunk(text=chunk_text, metadata={**meta}))
            overlap_segs = max(1, len(current) // 4)
            current = current[-overlap_segs:]
            current_len = sum(len(s) for s in current)
    if current:
        chunks.append(Chunk(text=" ".join(current), metadata={**meta}))
    return chunks


def chunk_text(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: dict | None = None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[Chunk]:
    """Unified entry: recursive | structure_aware | semantic."""
    if strategy == "structure_aware":
        return chunk_structure_aware(text, chunk_size, overlap, metadata)
    if strategy == "semantic" and embed_fn:
        return chunk_semantic(text, embed_fn, chunk_size, overlap, metadata, similarity_threshold=0.7)
    return chunk_recursive(text, chunk_size, overlap, metadata)

