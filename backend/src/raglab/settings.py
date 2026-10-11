"""Configuration, read once at startup and passed down. No module-level instance."""

from __future__ import annotations

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    database_url: str = "postgresql+asyncpg://localhost:5432/raglab"

    # Embeddings. Both options produce 384-dim vectors, so switching needs no migration,
    # only a re-ingest (chunks remember which model embedded them).
    embedding_provider: Literal["local", "openai"] = "local"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 384

    # Generators. CLIs are used when found on PATH; APIs when a key is set.
    claude_cli: str = "claude"
    codex_cli: str = "codex"
    cli_timeout_seconds: float = 300
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5-mini"

    # Worker
    embedded_worker: bool = True  # run the ingestion worker inside the API process
    worker_poll_seconds: float = 1.0
    max_upload_mb: int = 50

    cors_origins: list[str] = ["http://localhost:5173"]
