"""Add event data schema and tables

Revision ID: 9675352ad712
Revises: ad9b05c7d195
Create Date: 2025-07-31 19:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '9675352ad712'
down_revision: Union[str, None] = 'ad9b05c7d195'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create events schema
    op.execute('CREATE SCHEMA IF NOT EXISTS events')
    
    # Create event_data table
    op.create_table(
        'event_data',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('event_datetime', sa.DateTime(timezone=True), nullable=False),
        sa.Column('event_name', sa.String(length=500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('impact', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('symbol', sa.String(length=50), nullable=True),
        sa.Column('currency', sa.String(length=10), nullable=True),
        sa.Column('country', sa.String(length=10), nullable=True),
        sa.Column('source', sa.String(length=100), nullable=False),
        sa.Column('source_id', sa.String(length=255), nullable=True),
        sa.Column('event_data', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source', 'source_id', name='uq_source_event'),
        schema='events'
    )
    
    # Create indexes for event_data
    op.create_index('idx_event_datetime', 'event_data', ['event_datetime'], schema='events')
    op.create_index('idx_event_type_datetime', 'event_data', ['event_type', 'event_datetime'], schema='events')
    op.create_index('idx_symbol_datetime', 'event_data', ['symbol', 'event_datetime'], schema='events')
    op.create_index('idx_currency_datetime', 'event_data', ['currency', 'event_datetime'], schema='events')
    
    # Create economic_events table
    op.create_table(
        'economic_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('event_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('event_datetime', sa.DateTime(timezone=True), nullable=False),
        sa.Column('currency', sa.String(length=10), nullable=True),
        sa.Column('actual', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('forecast', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('previous', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('revised', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('unit', sa.String(length=50), nullable=True),
        sa.Column('frequency', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['event_id'], ['events.event_data.id'], ),
        sa.PrimaryKeyConstraint('id'),
        schema='events'
    )
    
    # Create indexes for economic_events
    op.create_index('idx_econ_event_id', 'economic_events', ['event_id'], schema='events')
    op.create_index('idx_econ_datetime', 'economic_events', ['event_datetime'], schema='events')
    
    # Create earnings_events table
    op.create_table(
        'earnings_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('event_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('symbol', sa.String(length=50), nullable=False),
        sa.Column('event_datetime', sa.DateTime(timezone=True), nullable=False),
        sa.Column('eps_actual', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('eps_estimate', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('eps_surprise', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('eps_surprise_pct', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('revenue_actual', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('revenue_estimate', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('revenue_surprise', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('revenue_surprise_pct', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('guidance', sa.Text(), nullable=True),
        sa.Column('call_time', sa.String(length=10), nullable=True),
        sa.Column('fiscal_period', sa.String(length=20), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['event_id'], ['events.event_data.id'], ),
        sa.PrimaryKeyConstraint('id'),
        schema='events'
    )
    
    # Create indexes for earnings_events
    op.create_index('idx_earn_event_id', 'earnings_events', ['event_id'], schema='events')
    op.create_index('idx_earn_symbol_datetime', 'earnings_events', ['symbol', 'event_datetime'], schema='events')


def downgrade() -> None:
    # Drop tables
    op.drop_table('earnings_events', schema='events')
    op.drop_table('economic_events', schema='events')
    op.drop_table('event_data', schema='events')
    
    # Drop schema
    op.execute('DROP SCHEMA IF EXISTS events CASCADE')