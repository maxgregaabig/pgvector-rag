"""A deterministic provider used by tests and by offline eval runs.

Embeddings come from the hashing trick: every word is hashed to one dimension
with a stable sign, and the resulting vector is L2 normalised. Two texts that
share words end up with a high cosine similarity, which is enough to exercise
the retrieval path end to end. It is a lexical model, not a semantic one, and
the README says so.

The class counts its calls so a test can assert that a cache hit really did
skip the provider.
"""

from __future__ import annotations

import hashlib
import math
import re

from app.providers.base import ChatResult, EmbeddingResult, estimate_tokens

_WORD = re.compile(r"[a-z0-9]+")
_CONTEXT_BLOCK = re.compile(
    r"^\[(?P<marker>\d+)\]\s+source=\S+\n(?P<body>.*?)(?=\n\n\[\d+\]\s+source=|\Z)",
    re.MULTILINE | re.DOTALL,
)
_FIRST_SENTENCE = re.compile(r"^.*?[.!?](?=\s|$)")


NGRAM_SIZE = 4
NGRAM_WEIGHT = 0.35


def _add_feature(vector: list[float], feature: str, weight: float) -> None:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    index = int.from_bytes(digest[:4], "big") % len(vector)
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    vector[index] += sign * weight


def hash_embedding(text: str, dim: int) -> list[float]:
    """Map text to a unit length vector of size dim.

    Whole words carry most of the weight. Character n-grams of each word are
    added at a lower weight so that plurals and other small spelling
    differences still overlap.
    """
    vector = [0.0] * dim
    for word in _WORD.findall(text.lower()):
        _add_feature(vector, f"w:{word}", 1.0)
        padded = f"^{word}$"
        for start in range(max(len(padded) - NGRAM_SIZE + 1, 1)):
            _add_feature(vector, f"g:{padded[start : start + NGRAM_SIZE]}", NGRAM_WEIGHT)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        # Empty or symbol-only text still needs a valid unit vector.
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


class FakeProvider:
    """In-process stand-in for the OpenAI provider."""

    name = "fake"

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim
        self.embed_calls = 0
        self.chat_calls = 0
        self.embedded_texts = 0

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        self.embed_calls += 1
        self.embedded_texts += len(texts)
        return EmbeddingResult(
            vectors=[hash_embedding(text, self.dim) for text in texts],
            tokens=sum(estimate_tokens(text) for text in texts),
        )

    async def chat(self, system: str, user: str) -> ChatResult:
        self.chat_calls += 1
        answer = _summarise_context(user)
        return ChatResult(
            text=answer,
            input_tokens=estimate_tokens(system) + estimate_tokens(user),
            output_tokens=estimate_tokens(answer),
        )

    async def aclose(self) -> None:
        return None


def _summarise_context(user_prompt: str) -> str:
    """Build an answer out of the context blocks in the prompt.

    The real model is asked to cite sources as [1], [2] and so on. The fake
    does the same thing mechanically so the citation parsing in the service
    layer is exercised by tests.
    """
    blocks = list(_CONTEXT_BLOCK.finditer(user_prompt))
    if not blocks:
        return "I could not find anything relevant in the indexed documents."
    parts = []
    for block in blocks[:3]:
        body = block.group("body").strip()
        match = _FIRST_SENTENCE.match(body)
        sentence = (match.group(0) if match else body[:200]).strip()
        parts.append(f"{sentence} [{block.group('marker')}]")
    return "Based on the indexed documents: " + " ".join(parts)
