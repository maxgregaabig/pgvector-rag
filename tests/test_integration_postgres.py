"""Tests that need a real PostgreSQL with the pgvector extension.

They are skipped with a message when DATABASE_URL is not set. Bring the
services up with 'docker compose up -d postgres redis' first, or run
'make test-integration'.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.cache import MemoryCache
from app.config import Settings
from app.models import ChunkRecord, DocumentRecord
from app.providers.fake import FakeProvider, hash_embedding
from app.service import RagService
from app.store.postgres import PostgresVectorStore
from tests.conftest import FINTECH_FAQ, require_env

DIM = 1536


@pytest.fixture
async def store():
    dsn = require_env("DATABASE_URL", "PostgreSQL")
    store = PostgresVectorStore(dsn=dsn, run_migrations=True)
    await store.setup()
    async with store.pool.acquire() as connection:
        await connection.execute("TRUNCATE documents CASCADE")
    try:
        yield store
    finally:
        async with store.pool.acquire() as connection:
            await connection.execute("TRUNCATE documents CASCADE")
        await store.aclose()


@pytest.fixture
def pg_settings() -> Settings:
    return Settings(
        store_backend="postgres",
        cache_backend="memory",
        provider="fake",
        embedding_dim=DIM,
        chunk_target_chars=200,
        chunk_overlap_chars=40,
        default_top_k=3,
    )


def _document(source: str = "faq.md") -> DocumentRecord:
    return DocumentRecord(id=uuid4(), source=source, title="FAQ", metadata={"team": "payments"})


def _chunk(document_id, ordinal: int, body: str) -> ChunkRecord:
    return ChunkRecord(
        id=uuid4(),
        document_id=document_id,
        ordinal=ordinal,
        content=body,
        content_hash=f"hash-{ordinal}",
        token_estimate=len(body) // 4,
        embedding=hash_embedding(body, DIM),
    )


async def test_migrations_create_the_vector_index(store: PostgresVectorStore):
    async with store.pool.acquire() as connection:
        extension = await connection.fetchval(
            "SELECT count(*) FROM pg_extension WHERE extname = 'vector'"
        )
        index = await connection.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_hnsw_idx'"
        )
    assert extension == 1
    assert "hnsw" in index
    assert "vector_cosine_ops" in index


async def test_migrations_are_idempotent(store: PostgresVectorStore):
    await store.migrate()
    async with store.pool.acquire() as connection:
        applied = await connection.fetchval("SELECT count(*) FROM schema_migrations")
    assert applied >= 1


async def test_roundtrip_through_pgvector(store: PostgresVectorStore):
    document = _document()
    bodies = [
        "Chargebacks can be raised within 120 days of the transaction date.",
        "The daily payout cutoff is 16:00 UTC.",
        "Every ledger entry is immutable and corrected by a reversing entry.",
    ]
    chunks = [_chunk(document.id, index, body) for index, body in enumerate(bodies)]
    await store.add_document(document, chunks)
    assert await store.count_chunks() == 3

    hits = await store.search(hash_embedding("payout cutoff time", DIM), top_k=2)
    assert len(hits) == 2
    assert "payout cutoff" in hits[0].content
    assert hits[0].score > hits[1].score
    assert hits[0].source == "faq.md"
    assert hits[0].title == "FAQ"

    assert await store.delete_document(document.id) is True
    assert await store.count_chunks() == 0
    assert await store.delete_document(document.id) is False


async def test_search_can_be_scoped_to_document_ids(store: PostgresVectorStore):
    first, second = _document("a.md"), _document("b.md")
    await store.add_document(first, [_chunk(first.id, 0, "Settlement timing rules for cards.")])
    await store.add_document(second, [_chunk(second.id, 0, "Settlement timing rules for cards.")])

    scoped = await store.search(hash_embedding("settlement timing", DIM), 5, [second.id])
    assert {hit.document_id for hit in scoped} == {second.id}
    assert len(await store.search(hash_embedding("settlement timing", DIM), 5)) == 2


async def test_service_end_to_end_against_postgres(store: PostgresVectorStore, pg_settings):
    service = RagService(pg_settings, store, MemoryCache(), FakeProvider(dim=DIM))
    ingested = await service.ingest(source="faq.md", text=FINTECH_FAQ)
    assert ingested.chunks > 1

    answer = await service.ask("How long is the chargeback window?", top_k=3)
    assert answer.citations
    assert all(c.document_id == ingested.document_id for c in answer.citations)
