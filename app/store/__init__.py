"""Vector store backends."""

from app.store.base import VectorStore, to_vector_literal
from app.store.memory import MemoryVectorStore, cosine_similarity

__all__ = [
    "MemoryVectorStore",
    "VectorStore",
    "build_store",
    "cosine_similarity",
    "to_vector_literal",
]


def build_store(settings) -> VectorStore:
    """Create the store named by the settings."""
    if settings.store_backend == "memory":
        return MemoryVectorStore()
    from app.store.postgres import PostgresVectorStore

    return PostgresVectorStore(
        dsn=settings.database_url,
        run_migrations=settings.run_migrations_on_startup,
        migrations_dir=settings.migrations_dir,
    )
