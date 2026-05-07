"""Pure-Python text utilities — kept dependency-free so smoke tests can import them."""

from __future__ import annotations


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Approximate token-based chunking via word counting.

    Args:
        text: Document text.
        size: Target chunk size in words.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of chunks. Empty list if input is empty.
    """
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size])
        chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks
