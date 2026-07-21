"""PostgreSQL and pgvector store.

Queries go through asyncpg directly. Embeddings are sent as pgvector text
literals and cast in SQL, which keeps the driver free of custom type codecs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import UUID

import asyncpg

from app.models import ChunkRecord, DocumentRecord, SearchHit
from app.store.base import to_vector_literal

logger = logging.getLogger(__name__)

DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"

_SEARCH_SQL = """
SELECT c.id, c.document_id, c.ordinal, c.content,
       1 - (c.embedding <=> $1::vector) AS score,
       d.source, d.title
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE ($3::uuid[] IS NULL OR c.document_id = ANY ($3::uuid[]))
ORDER BY c.embedding <=> $1::vector
LIMIT $2
"""


class PostgresVectorStore:
    name = "postgres"

    def __init__(
        self,
        dsn: str,
        run_migrations: bool = True,
        migrations_dir: Path | str | None = None,
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._run_migrations = run_migrations
        self._migrations_dir = Path(migrations_dir) if migrations_dir else DEFAULT_MIGRATIONS_DIR
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("store is not set up, call setup() first")
        return self._pool

    async def setup(self) -> None:
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn, min_size=self._min_size, max_size=self._max_size
        )
        if self._run_migrations:
            await self.migrate()

    async def migrate(self) -> None:
        """Apply every .sql file in migrations/ that has not run yet."""
        files = sorted(self._migrations_dir.glob("*.sql"))
        if not files:
            raise RuntimeError(f"no migrations found in {self._migrations_dir}")
        async with self.pool.acquire() as connection:
            await connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " name TEXT PRIMARY KEY,"
                " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            applied = {
                row["name"] for row in await connection.fetch("SELECT name FROM schema_migrations")
            }
            for path in files:
                if path.name in applied:
                    continue
                logger.info("applying migration %s", path.name)
                async with connection.transaction():
                    await connection.execute(path.read_text(encoding="utf-8"))
                    await connection.execute(
                        "INSERT INTO schema_migrations (name) VALUES ($1)", path.name
                    )

    async def add_document(self, document: DocumentRecord, chunks: list[ChunkRecord]) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "INSERT INTO documents (id, source, title, metadata)"
                " VALUES ($1, $2, $3, $4::jsonb)",
                document.id,
                document.source,
                document.title,
                json.dumps(document.metadata),
            )
            await connection.executemany(
                "INSERT INTO chunks"
                " (id, document_id, ordinal, content, content_hash, token_estimate, embedding)"
                " VALUES ($1, $2, $3, $4, $5, $6, $7::vector)",
                [
                    (
                        chunk.id,
                        document.id,
                        chunk.ordinal,
                        chunk.content,
                        chunk.content_hash,
                        chunk.token_estimate,
                        to_vector_literal(chunk.embedding),
                    )
                    for chunk in chunks
                ],
            )

    async def delete_document(self, document_id: UUID) -> bool:
        async with self.pool.acquire() as connection:
            result = await connection.execute("DELETE FROM documents WHERE id = $1", document_id)
        return result.rsplit(" ", 1)[-1] != "0"

    async def search(
        self,
        embedding: list[float],
        top_k: int,
        document_ids: list[UUID] | None = None,
    ) -> list[SearchHit]:
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                _SEARCH_SQL,
                to_vector_literal(embedding),
                top_k,
                list(document_ids) if document_ids else None,
            )
        return [
            SearchHit(
                chunk_id=row["id"],
                document_id=row["document_id"],
                source=row["source"],
                title=row["title"],
                ordinal=row["ordinal"],
                content=row["content"],
                score=float(row["score"]),
            )
            for row in rows
        ]

    async def count_chunks(self) -> int:
        async with self.pool.acquire() as connection:
            return int(await connection.fetchval("SELECT count(*) FROM chunks"))

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
