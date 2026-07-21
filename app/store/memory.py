"""In-process vector store.

Brute force cosine similarity over a dict. It exists so the API, the tests and
the eval script can run without PostgreSQL. It is not meant for real data
volumes: search is O(number of chunks).
"""

from __future__ import annotations

import math
from uuid import UUID

from app.models import ChunkRecord, DocumentRecord, SearchHit


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class MemoryVectorStore:
    name = "memory"

    def __init__(self) -> None:
        self._documents: dict[UUID, DocumentRecord] = {}
        self._chunks: dict[UUID, list[ChunkRecord]] = {}

    async def setup(self) -> None:
        return None

    async def add_document(self, document: DocumentRecord, chunks: list[ChunkRecord]) -> None:
        self._documents[document.id] = document
        self._chunks[document.id] = list(chunks)

    async def delete_document(self, document_id: UUID) -> bool:
        self._chunks.pop(document_id, None)
        return self._documents.pop(document_id, None) is not None

    async def search(
        self,
        embedding: list[float],
        top_k: int,
        document_ids: list[UUID] | None = None,
    ) -> list[SearchHit]:
        wanted = set(document_ids) if document_ids else None
        hits: list[SearchHit] = []
        for document_id, chunks in self._chunks.items():
            if wanted is not None and document_id not in wanted:
                continue
            document = self._documents[document_id]
            for chunk in chunks:
                hits.append(
                    SearchHit(
                        chunk_id=chunk.id,
                        document_id=document_id,
                        source=document.source,
                        title=document.title,
                        ordinal=chunk.ordinal,
                        content=chunk.content,
                        score=cosine_similarity(embedding, chunk.embedding),
                    )
                )
        hits.sort(key=lambda hit: (-hit.score, str(hit.chunk_id)))
        return hits[:top_k]

    async def count_chunks(self) -> int:
        return sum(len(chunks) for chunks in self._chunks.values())

    async def aclose(self) -> None:
        return None
