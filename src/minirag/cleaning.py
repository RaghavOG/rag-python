"""Text cleaning & normalization. Garbage in → hallucination out."""
import re

try:
    from langdetect import detect, LangDetectException
except ImportError:
    detect = None
    LangDetectException = Exception


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip."""
    return re.sub(r"\s+", " ", text).strip()


def remove_header_footer_candidates(text: str, min_line_len: int = 10) -> str:
    """Remove lines that look like headers/footers (very short, repeated at top/bottom)."""
    lines = text.splitlines()
    if len(lines) < 5:
        return text

    def is_likely_header_footer(line: str) -> bool:
        s = line.strip()
        if len(s) < min_line_len:
            return True
        if re.match(r"^[\d\s\-\.\/]+$", s):  # page numbers, dates
            return True
        return False

    start = 0
    while start < len(lines) and is_likely_header_footer(lines[start]):
        start += 1
    end = len(lines)
    while end > start and is_likely_header_footer(lines[end - 1]):
        end -= 1
    return "\n".join(lines[start:end])


def deduplicate_sentences(text: str) -> str:
    """Remove consecutive duplicate sentences (and near-duplicates by line)."""
    lines = [normalize_whitespace(line) for line in text.splitlines() if line.strip()]
    seen = set()
    out = []
    for line in lines:
        key = line.lower()[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return "\n".join(out)


def preserve_blocks(text: str) -> str:
    """Normalize whitespace but preserve code blocks and tables (markdown-style)."""
    out = []
    in_code = False
    for part in re.split(r"(```[\w]*\n?|```)", text):
        if part.startswith("```"):
            in_code = not in_code
            out.append(part)
            continue
        if in_code:
            out.append(part)
            continue
        out.append(normalize_whitespace(part))
    return "".join(out) if out else text


def detect_language(text: str) -> str | None:
    """Return ISO language code or None if detection fails."""
    if not detect:
        return None
    try:
        sample = text[:2000] if len(text) > 2000 else text
        return detect(sample)
    except LangDetectException:
        return None


def clean_document(
    text: str,
    *,
    normalize_ws: bool = True,
    remove_headers_footers: bool = True,
    dedupe: bool = True,
    preserve_code_tables: bool = True,
    min_lang_length: int = 50,
) -> str:
    """Full cleaning pipeline. Preserve code/tables; optionally skip non-English if desired."""
    if normalize_ws and not preserve_code_tables:
        text = normalize_whitespace(text)
    elif preserve_code_tables:
        text = preserve_blocks(text)
    if remove_headers_footers:
        text = remove_header_footer_candidates(text)
    if dedupe:
        text = deduplicate_sentences(text)
    if normalize_ws and preserve_code_tables:
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r" +", " ", text)
    return text.strip()

