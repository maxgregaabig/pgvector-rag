"""Application settings, read from the environment."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Backends. The "memory" and "fake" variants exist so the API and the eval
    # script can run with no database, no Redis and no network.
    store_backend: Literal["postgres", "memory"] = "postgres"
    cache_backend: Literal["redis", "memory"] = "redis"
    provider: Literal["openai", "fake"] = "fake"

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    redis_url: str = "redis://localhost:6379/0"
    run_migrations_on_startup: bool = True
    # Where the .sql files live. Set explicitly in the container image, where
    # the package is installed and the repository layout is not present.
    migrations_dir: str | None = None

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    embedding_dim: int = 1536
    request_timeout_seconds: float = 30.0

    chunk_target_chars: int = 900
    chunk_overlap_chars: int = 150
    default_top_k: int = 5
    max_top_k: int = 50

    answer_cache_ttl_seconds: int = 3600
    embedding_cache_ttl_seconds: int = 604800

    # Prices in USD per one million tokens. Used by the eval script to turn
    # token counts into a cost estimate. Override them if your rates differ.
    embedding_price_per_mtok: float = 0.02
    chat_input_price_per_mtok: float = 0.15
    chat_output_price_per_mtok: float = 0.60
