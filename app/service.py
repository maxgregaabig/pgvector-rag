"""Ingestion, search and question answering."""

from __future__ import annotations

import json
import logging
import re
import time
from uuid import UUID, uuid4

from app.cache import Cache, answer_key, content_hash, embedding_key
from app.chunking import chunk_text
from app.config import Settings
from app.models import (
    AskResponse,
    ChunkRecord,
    Citation,
    DocumentRecord,
    IngestResponse,
    SearchHit,
    SearchResponse,
    SearchResult,
    Usage,
)
from app.providers.base import Provider, estimate_tokens
from app.store.base import VectorStore

logger = logging.getLogger(__name__)

_MARKER = re.compile(r"\[(\d{1,3})\]")

SYSTEM_PROMPT = (
    "You answer questions about a document collection. Use only the context "
    "blocks given to you. Cite every claim with the bracketed number of the "
    "block it came from, for example [1]. If the context does not answer the "
    "question, say so instead of guessing."
)


class EmptyDocumentError(ValueError):
    """Raised when a document contains no usable text."""


class RagService:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        cache: Cache,
        provider: Provider,
    ) -> None:
        self.settings = settings
        self.store = store
        self.cache = cache
        self.provider = provider

    async def ingest(
        self,
        source: str,
        text: str,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> IngestResponse:
        chunks = chunk_text(
            text,
            target_chars=self.settings.chunk_target_chars,
            overlap_chars=self.settings.chunk_overlap_chars,
        )
        if not chunks:
            raise EmptyDocumentError("document produced no chunks")

        vectors, tokens, cache_hits = await self._embed_cached(chunks)
        document = DocumentRecord(id=uuid4(), source=source, title=title, metadata=metadata or {})
        records = [
            ChunkRecord(
                id=uuid4(),
                document_id=document.id,
                ordinal=ordinal,
                content=body,
                content_hash=content_hash(body),
                token_estimate=estimate_tokens(body),
                embedding=vector,
            )
            for ordinal, (body, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        await self.store.add_document(document, records)
        return IngestResponse(
            document_id=document.id,
            chunks=len(records),
            embedding_tokens=tokens,
            cached_embeddings=cache_hits,
        )

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[UUID] | None = None,
    ) -> SearchResponse:
        started = time.perf_counter()
        k = self._resolve_top_k(top_k)
        vectors, _, _ = await self._embed_cached([query])
        hits = await self.store.search(vectors[0], k, document_ids)
        return SearchResponse(
            query=query,
            results=[
                SearchResult(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    source=hit.source,
                    title=hit.title,
                    ordinal=hit.ordinal,
                    content=hit.content,
                    score=round(hit.score, 6),
                )
                for hit in hits
            ],
            took_ms=_elapsed_ms(started),
        )

    async def ask(
        self,
        question: str,
        top_k: int | None = None,
        document_ids: list[UUID] | None = None,
    ) -> AskResponse:
        started = time.perf_counter()
        k = self._resolve_top_k(top_k)
        key = answer_key(self.settings.chat_model, question, k, document_ids)

        cached = await self.cache.get(key)
        if cached is not None:
            payload = json.loads(cached)
            return AskResponse(
                question=question,
                answer=payload["answer"],
                citations=[Citation(**item) for item in payload["citations"]],
                cached=True,
                usage=Usage(**payload["usage"]),
                took_ms=_elapsed_ms(started),
            )

        vectors, embedding_tokens, _ = await self._embed_cached([question])
        hits = await self.store.search(vectors[0], k, document_ids)
        prompt = build_user_prompt(question, hits)
        result = await self.provider.chat(SYSTEM_PROMPT, prompt)
        citations = extract_citations(result.text, hits)
        usage = Usage(
            embedding_tokens=embedding_tokens,
            chat_input_tokens=result.input_tokens,
            chat_output_tokens=result.output_tokens,
        )
        await self.cache.set(
            key,
            json.dumps(
                {
                    "answer": result.text,
                    "citations": [json.loads(c.model_dump_json()) for c in citations],
                    "usage": usage.model_dump(),
                }
            ),
            self.settings.answer_cache_ttl_seconds,
        )
        return AskResponse(
            question=question,
            answer=result.text,
            citations=citations,
            cached=False,
            usage=usage,
            took_ms=_elapsed_ms(started),
        )

    async def _embed_cached(self, texts: list[str]) -> tuple[list[list[float]], int, int]:
        """Embed texts, reusing cached vectors.

        Returns the vectors, the tokens actually billed by the provider, and
        how many vectors came from the cache.
        """
        model = self.settings.embedding_model
        dim = self.settings.embedding_dim
        vectors: list[list[float] | None] = [None] * len(texts)
        misses: list[int] = []

        for index, text in enumerate(texts):
            raw = await self.cache.get(embedding_key(model, dim, text))
            if raw is None:
                misses.append(index)
            else:
                vectors[index] = json.loads(raw)

        tokens = 0
        if misses:
            result = await self.provider.embed([texts[index] for index in misses])
            tokens = result.tokens
            for index, vector in zip(misses, result.vectors, strict=True):
                vectors[index] = vector
                await self.cache.set(
                    embedding_key(model, dim, texts[index]),
                    json.dumps(vector),
                    self.settings.embedding_cache_ttl_seconds,
                )

        return (
            [vector for vector in vectors if vector is not None],
            tokens,
            len(texts) - len(misses),
        )

    def _resolve_top_k(self, top_k: int | None) -> int:
        if top_k is None:
            return self.settings.default_top_k
        return max(1, min(top_k, self.settings.max_top_k))


def build_user_prompt(question: str, hits: list[SearchHit]) -> str:
    """Render the retrieved chunks as numbered, citable context blocks."""
    if not hits:
        return f"Question: {question}\n\nContext:\n(no matching documents)"
    blocks = [
        f"[{index}] source={hit.source}#{hit.ordinal}\n{hit.content}"
        for index, hit in enumerate(hits, start=1)
    ]
    return f"Question: {question}\n\nContext:\n" + "\n\n".join(blocks)


def extract_citations(answer: str, hits: list[SearchHit]) -> list[Citation]:
    """Map the [n] markers the model used back to the chunks they refer to.

    Markers outside the range of retrieved chunks are dropped, so a
    hallucinated citation never reaches the client.
    """
    citations: list[Citation] = []
    seen: set[int] = set()
    for match in _MARKER.finditer(answer):
        marker = int(match.group(1))
        if marker in seen or not 1 <= marker <= len(hits):
            continue
        seen.add(marker)
        hit = hits[marker - 1]
        citations.append(
            Citation(
                marker=marker,
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                source=hit.source,
                ordinal=hit.ordinal,
                score=round(hit.score, 6),
            )
        )
    return citations


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
