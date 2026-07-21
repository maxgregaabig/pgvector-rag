"""Tests that need a real Redis.

Skipped with a message when REDIS_URL is not set.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.cache import RedisCache
from app.config import Settings
from app.providers.fake import FakeProvider
from app.service import RagService
from app.store.memory import MemoryVectorStore
from tests.conftest import FINTECH_FAQ, require_env


@pytest.fixture
async def cache():
    url = require_env("REDIS_URL", "Redis")
    cache = RedisCache(url)
    try:
        yield cache
    finally:
        await cache.aclose()


async def test_redis_roundtrip_and_ttl(cache: RedisCache):
    key = f"test:{uuid4()}"
    assert await cache.get(key) is None
    await cache.set(key, "value", 60)
    assert await cache.get(key) == "value"
    ttl = await cache._client.ttl(key)
    assert 0 < ttl <= 60


async def test_answers_survive_a_new_service_instance(cache: RedisCache):
    settings = Settings(
        store_backend="memory",
        cache_backend="redis",
        provider="fake",
        embedding_dim=256,
        chunk_target_chars=200,
        chunk_overlap_chars=40,
        default_top_k=3,
        chat_model=f"test-model-{uuid4()}",
    )
    store = MemoryVectorStore()
    first_provider = FakeProvider(dim=settings.embedding_dim)
    first = RagService(settings, store, cache, first_provider)
    await first.ingest(source="faq.md", text=FINTECH_FAQ)
    original = await first.ask("How long is the chargeback window?", top_k=3)
    assert original.cached is False

    # A different process would build a new service and a new provider. The
    # answer must still come back from Redis without a provider call.
    second_provider = FakeProvider(dim=settings.embedding_dim)
    second = RagService(settings, store, cache, second_provider)
    repeat = await second.ask("How long is the chargeback window?", top_k=3)

    assert repeat.cached is True
    assert repeat.answer == original.answer
    assert second_provider.chat_calls == 0
    assert second_provider.embed_calls == 0
