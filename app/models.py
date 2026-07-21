"""Request and response schemas, plus the internal record types."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from pydantic import BaseModel, Field


@dataclass(slots=True)
class ChunkRecord:
    """One embedded chunk as it is stored."""

    id: UUID
    document_id: UUID
    ordinal: int
    content: str
    content_hash: str
    token_estimate: int
    embedding: list[float] = field(repr=False, default_factory=list)


@dataclass(slots=True)
class DocumentRecord:
    id: UUID
    source: str
    title: str | None
    metadata: dict


@dataclass(slots=True)
class SearchHit:
    chunk_id: UUID
    document_id: UUID
    source: str
    title: str | None
    ordinal: int
    content: str
    score: float


class IngestRequest(BaseModel):
    source: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1)
    title: str | None = Field(default=None, max_length=500)
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: UUID
    chunks: int
    embedding_tokens: int
    cached_embeddings: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[UUID] | None = None


class SearchResult(BaseModel):
    chunk_id: UUID
    document_id: UUID
    source: str
    title: str | None
    ordinal: int
    content: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    took_ms: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[UUID] | None = None


class Citation(BaseModel):
    marker: int
    chunk_id: UUID
    document_id: UUID
    source: str
    ordinal: int
    score: float


class Usage(BaseModel):
    embedding_tokens: int = 0
    chat_input_tokens: int = 0
    chat_output_tokens: int = 0


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    cached: bool
    usage: Usage
    took_ms: float


class HealthResponse(BaseModel):
    status: str
    store: str
    cache: str
    provider: str
