"""The vector store interface."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.models import ChunkRecord, DocumentRecord, SearchHit


class VectorStore(Protocol):
    name: str

    async def setup(self) -> None:
        """Open connections and apply migrations if the backend needs them."""

    async def add_document(self, document: DocumentRecord, chunks: list[ChunkRecord]) -> None: ...

    async def delete_document(self, document_id: UUID) -> bool:
        """Return True if a document was removed."""

    async def search(
        self,
        embedding: list[float],
        top_k: int,
        document_ids: list[UUID] | None = None,
    ) -> list[SearchHit]:
        """Nearest neighbours by cosine similarity, best first."""

    async def count_chunks(self) -> int: ...

    async def aclose(self) -> None: ...


def to_vector_literal(embedding: list[float]) -> str:
    """Format a Python list the way pgvector parses it."""
    return "[" + ",".join(repr(float(value)) for value in embedding) + "]"
