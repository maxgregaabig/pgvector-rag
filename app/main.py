"""FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status

from app.cache import Cache, build_cache
from app.config import Settings
from app.models import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from app.providers import build_provider
from app.providers.base import Provider
from app.service import EmptyDocumentError, RagService
from app.store import build_store
from app.store.base import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@dataclass
class Container:
    """The wired up objects for one application instance."""

    settings: Settings
    store: VectorStore
    cache: Cache
    provider: Provider
    service: RagService


def build_container(settings: Settings) -> Container:
    store = build_store(settings)
    cache = build_cache(settings)
    provider = build_provider(settings)
    return Container(
        settings=settings,
        store=store,
        cache=cache,
        provider=provider,
        service=RagService(settings, store, cache, provider),
    )


def get_service(request: Request) -> RagService:
    return request.app.state.container.service


# FastAPI's dependency, written as an annotation so the callable is not a
# mutable default argument.
ServiceDep = Annotated[RagService, Depends(get_service)]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = build_container(settings)
        await container.store.setup()
        app.state.container = container
        try:
            yield
        finally:
            await container.store.aclose()
            await container.cache.aclose()
            await container.provider.aclose()

    app = FastAPI(
        title="pgvector-rag",
        version="0.1.0",
        summary="Document ingestion, vector search and grounded answers.",
        lifespan=lifespan,
    )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz(request: Request) -> HealthResponse:
        container: Container = request.app.state.container
        return HealthResponse(
            status="ok",
            store=container.store.name,
            cache=container.cache.name,
            provider=container.provider.name,
        )

    @app.post("/v1/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
    async def ingest(payload: IngestRequest, service: ServiceDep) -> IngestResponse:
        try:
            return await service.ingest(
                source=payload.source,
                text=payload.text,
                title=payload.title,
                metadata=payload.metadata,
            )
        except EmptyDocumentError as error:
            # 422 spelled out: the starlette constant for it was renamed and
            # the old name warns on newer versions.
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.delete("/v1/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_document(document_id: UUID, service: ServiceDep) -> Response:
        if not await service.store.delete_document(document_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/v1/search", response_model=SearchResponse)
    async def search(payload: SearchRequest, service: ServiceDep) -> SearchResponse:
        return await service.search(
            query=payload.query, top_k=payload.top_k, document_ids=payload.document_ids
        )

    @app.post("/v1/ask", response_model=AskResponse)
    async def ask(payload: AskRequest, service: ServiceDep) -> AskResponse:
        return await service.ask(
            question=payload.question, top_k=payload.top_k, document_ids=payload.document_ids
        )

    return app


app = create_app()
