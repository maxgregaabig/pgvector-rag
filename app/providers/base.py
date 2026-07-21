"""The provider interface.

Everything that talks to an embedding or chat model goes through this
protocol. The rest of the application never imports a concrete provider, which
is what lets the test suite run with a deterministic local implementation and
no network access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class EmbeddingResult:
    vectors: list[list[float]]
    tokens: int


@dataclass(slots=True)
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int


class Provider(Protocol):
    """Embedding and chat completion backend."""

    name: str
    dim: int

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts. The order of the output matches the input."""
        ...

    async def chat(self, system: str, user: str) -> ChatResult:
        """Run a single-turn chat completion."""
        ...

    async def aclose(self) -> None:
        """Release any network resources."""
        ...


def estimate_tokens(text: str) -> int:
    """Rough token count: about four characters per token.

    This avoids a tokenizer dependency that would need to download vocabulary
    files at runtime. It is an estimate and the eval script labels it as one.
    """
    return max(1, math.ceil(len(text) / 4)) if text else 0
