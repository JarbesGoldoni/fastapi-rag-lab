"""Configuration, read once at startup and passed down. No module-level instance."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env, wherever the process starts from (not the current directory).
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_prefix="", extra="ignore")

    database_url: str = "postgresql+asyncpg://localhost:5432/raglab"

    # Embeddings. Both options produce 384-dim vectors (the schema's size), so switching
    # needs no migration; documents embedded with another model are re-ingested at startup.
    embedding_provider: Literal["local", "openai"] = "local"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"

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
    worker_lease_seconds: float = 30  # a job whose worker stops renewing is retried after this
    worker_heartbeat_seconds: float = 2  # also how fast a deleted document's job stops
    worker_max_attempts: int = 3
    max_upload_mb: int = 50

    # Browsers may only send writes from these origins (or the backend's own origin), and
    # only these host names are served, which blocks DNS rebinding.
    cors_origins: list[str] = ["http://localhost:5173"]
    allowed_hosts: list[str] = ["localhost", "127.0.0.1"]
