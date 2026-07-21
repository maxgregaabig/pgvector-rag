"""Embedding and chat providers."""

from app.providers.base import ChatResult, EmbeddingResult, Provider, estimate_tokens
from app.providers.fake import FakeProvider, hash_embedding

__all__ = [
    "ChatResult",
    "EmbeddingResult",
    "FakeProvider",
    "Provider",
    "estimate_tokens",
    "hash_embedding",
]


def build_provider(settings) -> Provider:
    """Create the provider named by the settings."""
    if settings.provider == "fake":
        return FakeProvider(dim=settings.embedding_dim)
    # Imported here so the OpenAI client is only required when that provider is used.
    # pylint: disable=import-outside-toplevel
    from app.providers.openai import OpenAIProvider

    return OpenAIProvider(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
        dim=settings.embedding_dim,
        timeout=settings.request_timeout_seconds,
    )
