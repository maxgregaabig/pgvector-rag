"""Shared fixtures.

The default fixtures use the memory store, the memory cache and the fake
provider, so the whole default suite runs with no database, no Redis and no
network.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.cache import MemoryCache
from app.config import Settings
from app.main import create_app
from app.providers.fake import FakeProvider
from app.service import RagService
from app.store.memory import MemoryVectorStore

FINTECH_FAQ = """
Settlement of card payments happens on T plus two business days for domestic
transactions and T plus five for cross border transactions.

Chargebacks can be raised by the cardholder within 120 days of the transaction
date. The merchant has 20 days to respond with evidence.

The daily payout cutoff is 16:00 UTC. Payout batches created after the cutoff
are sent on the next banking day.

Every ledger entry is immutable. A correction is posted as a reversing entry
that references the original entry id.
"""


@pytest.fixture
def settings() -> Settings:
    return Settings(
        store_backend="memory",
        cache_backend="memory",
        provider="fake",
        embedding_dim=256,
        chunk_target_chars=200,
        chunk_overlap_chars=40,
        default_top_k=3,
    )


@pytest.fixture
def provider(settings: Settings) -> FakeProvider:
    return FakeProvider(dim=settings.embedding_dim)


@pytest.fixture
def service(settings: Settings, provider: FakeProvider) -> RagService:
    return RagService(settings, MemoryVectorStore(), MemoryCache(), provider)


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def fake_provider_of(client) -> FakeProvider:
    """The provider instance the running app is using."""
    return client.app.state.container.provider


def require_env(name: str, service_name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(
            f"{name} is not set, skipping {service_name} integration tests. "
            f"Run 'docker compose up -d postgres redis' and export {name} to enable them."
        )
    return value
