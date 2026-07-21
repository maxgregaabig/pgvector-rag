"""Cache layer.

Two things are cached, both keyed by a hash of their input:

* chunk embeddings, keyed by the embedding model plus the chunk text. Re-ingest
  of the same paragraph is free, which matters because ingestion is where the
  token spend is.
* answers, keyed by the chat model, the question, top_k and any document
  filter. A repeated question skips both the embedding call and the chat call.

Search results are deliberately not cached, see the README.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Protocol
from uuid import UUID


class Cache(Protocol):
    name: str

    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl_seconds: int) -> None: ...

    async def aclose(self) -> None: ...


class MemoryCache:
    """Process local cache with the same expiry behaviour as the Redis one.

    Both the TTL and the size bound matter. This is the default backend for the
    tests and the eval, and the keys are hashes of user questions and chunk
    bodies, so an unbounded dict would grow for the life of the process. Oldest
    entry is evicted once max_entries is reached.
    """

    name = "memory"

    def __init__(self, max_entries: int = 1024) -> None:
        self._data: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._max_entries = max_entries

    async def get(self, key: str) -> str | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at <= time.monotonic():
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._data[key] = (value, time.monotonic() + ttl_seconds)
        self._data.move_to_end(key)
        while len(self._data) > self._max_entries:
            self._data.popitem(last=False)

    async def aclose(self) -> None:
        self._data.clear()


class RedisCache:
    name = "redis"

    def __init__(self, url: str) -> None:
        # Imported here so the redis client is only required when this backend
        # is selected.
        # pylint: disable=import-outside-toplevel
        import redis.asyncio as redis

        self._client = redis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> str | None:
        return await self._client.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        await self._client.set(key, value, ex=ttl_seconds)

    async def aclose(self) -> None:
        await self._client.aclose()


def build_cache(settings) -> Cache:
    if settings.cache_backend == "memory":
        return MemoryCache()
    return RedisCache(settings.redis_url)


def _digest(*parts: str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def content_hash(text: str) -> str:
    """Stable hash of a chunk body, also stored on the chunk row."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_key(model: str, dim: int, text: str) -> str:
    return f"emb:v1:{model}:{dim}:{content_hash(text)}"


def answer_key(
    model: str,
    question: str,
    top_k: int,
    document_ids: list[UUID] | None,
) -> str:
    scope = json.dumps(sorted(str(value) for value in document_ids or []), separators=(",", ":"))
    return f"ans:v1:{model}:{top_k}:{_digest(question.strip().lower(), scope)}"
