"""Table definitions (SQLAlchemy Core). The Alembic migration creates the same schema."""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Column,
    Computed,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

EMBEDDING_DIMENSIONS = 384

metadata = MetaData()

documents = Table(
    "documents",
    metadata,
    Column("id", String(64), primary_key=True),  # sha256 of the file
    Column("filename", Text, nullable=False),
    Column("content", LargeBinary, nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("status", String(16), nullable=False),
    Column("pages", Integer),
    Column("chunks", Integer),
    Column("extractor", String(32)),
    Column("contextualized", String(32)),  # generator id used, if any
    Column("warnings", JSONB, nullable=False, server_default="[]"),
    Column("error", Text),
    Column("progress_stage", String(16)),
    Column("progress_done", Integer),
    Column("progress_total", Integer),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

chunks = Table(
    "chunks",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("document_id", String(64), ForeignKey("documents.id", ondelete="CASCADE"), index=True),
    Column("ordinal", Integer, nullable=False),
    Column("page", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("context", Text, nullable=False, server_default=""),
    Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=False),
    Column("embedding_model", String(128), nullable=False),
    Column(
        "tsv",
        TSVECTOR,
        Computed("to_tsvector('simple', context || ' ' || text)", persisted=True),
    ),
)

jobs = Table(
    "jobs",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("document_id", String(64), ForeignKey("documents.id", ondelete="CASCADE")),
    Column("options", JSONB, nullable=False, server_default="{}"),
    Column("status", String(16), nullable=False, server_default="pending"),
    Column("attempts", Integer, nullable=False, server_default="0"),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
