"""Extend audit_events for constitution Section 9 fields.

Revision ID: 20260626_audit_constitution_fields
Revises: 20260626_initial_audit_schema
Create Date: 2026-06-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260626_audit_constitution_fields"
down_revision: str | None = "20260626_initial_audit_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("audit_events", sa.Column("request_id", sa.String(length=64), nullable=True))
    op.add_column("audit_events", sa.Column("action", sa.String(length=64), nullable=False, server_default="unknown"))
    op.add_column("audit_events", sa.Column("ip_address", sa.String(length=45), nullable=True))
    op.add_column("audit_events", sa.Column("device", sa.String(length=128), nullable=True))
    op.add_column("audit_events", sa.Column("detail", sa.Text(), nullable=True))
    op.create_index("ix_audit_events_request_id", "audit_events", ["request_id"])
    op.create_index("ix_audit_events_action", "audit_events", ["action"])
    op.alter_column("audit_events", "action", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_audit_events_action", table_name="audit_events")
    op.drop_index("ix_audit_events_request_id", table_name="audit_events")
    op.drop_column("audit_events", "detail")
    op.drop_column("audit_events", "device")
    op.drop_column("audit_events", "ip_address")
    op.drop_column("audit_events", "action")
    op.drop_column("audit_events", "request_id")
