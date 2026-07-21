"""Text chunking.

The strategy is sentence packing: split the text into sentences, then fill a
chunk until it reaches the target size. Consecutive chunks share a tail of
whole sentences so a fact that sits on a chunk boundary still appears intact in
one of them. Sentences longer than the target are hard split on whitespace.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_WHITESPACE = re.compile(r"[ \t]+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units. Empty pieces are dropped."""
    parts = _SENTENCE_END.split(text.strip())
    return [_WHITESPACE.sub(" ", p).strip() for p in parts if p and p.strip()]


def _hard_split(sentence: str, limit: int) -> list[str]:
    """Break an oversized sentence on word boundaries."""
    words = sentence.split(" ")
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and size + addition > limit:
            pieces.append(" ".join(current))
            current = [word]
            size = len(word)
        else:
            current.append(word)
            size += addition
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(text: str, target_chars: int = 900, overlap_chars: int = 150) -> list[str]:
    """Split text into overlapping chunks of roughly target_chars characters.

    Returns an empty list for blank input. The overlap is measured in
    characters but applied in whole sentences, so a chunk never starts in the
    middle of a sentence.
    """
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if not 0 <= overlap_chars < target_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than target_chars")

    sentences: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) > target_chars:
            sentences.extend(_hard_split(sentence, target_chars))
        else:
            sentences.append(sentence)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for sentence in sentences:
        addition = len(sentence) + (1 if current else 0)
        if current and size + addition > target_chars:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap_chars)
            size = sum(len(s) for s in current) + max(len(current) - 1, 0)
            addition = len(sentence) + (1 if current else 0)
        current.append(sentence)
        size += addition

    if current:
        chunks.append(" ".join(current))
    return chunks


def _overlap_tail(sentences: list[str], overlap_chars: int) -> list[str]:
    """Return the trailing sentences that fit inside overlap_chars."""
    if overlap_chars <= 0:
        return []
    tail: list[str] = []
    size = 0
    for sentence in reversed(sentences):
        addition = len(sentence) + (1 if tail else 0)
        if size + addition > overlap_chars:
            break
        tail.insert(0, sentence)
        size += addition
    # Never carry the whole chunk forward, that would not terminate.
    if len(tail) >= len(sentences):
        tail = tail[1:]
    return tail
