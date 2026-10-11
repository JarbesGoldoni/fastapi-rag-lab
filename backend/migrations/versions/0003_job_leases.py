"""Job leases: a running job belongs to one worker until its lease expires.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("worker_id", sa.Text))
    op.add_column("jobs", sa.Column("lease_until", sa.DateTime(timezone=True)))
    # Jobs left running by the old code have no owner; let the next worker retry them.
    op.execute("UPDATE jobs SET lease_until = now() WHERE status = 'running'")


def downgrade() -> None:
    op.drop_column("jobs", "lease_until")
    op.drop_column("jobs", "worker_id")
