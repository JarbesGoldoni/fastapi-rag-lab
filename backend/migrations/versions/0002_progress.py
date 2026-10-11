"""Ingestion progress on documents: current stage and its measured done/total counts.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("progress_stage", sa.String(16)))
    op.add_column("documents", sa.Column("progress_done", sa.Integer))
    op.add_column("documents", sa.Column("progress_total", sa.Integer))


def downgrade() -> None:
    op.drop_column("documents", "progress_total")
    op.drop_column("documents", "progress_done")
    op.drop_column("documents", "progress_stage")
