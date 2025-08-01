"""Add economic data schema and tables

Revision ID: ad9b05c7d195
Revises: 153894776388
Create Date: 2025-07-31 18:16:09.708787

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ad9b05c7d195'
down_revision = '153894776388'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create economic data schema
    op.execute("CREATE SCHEMA IF NOT EXISTS economic_data")
    
    # Create economic_data table
    op.create_table(
        'economic_data',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('symbol', sa.String(50), nullable=False),
        sa.Column('date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('value', sa.Numeric(20, 8), nullable=False),
        sa.Column('revision_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_preliminary', sa.Boolean(), nullable=False, default=False),
        sa.Column('source', sa.String(50), nullable=False, default='FRED'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'date', name='uq_econ_symbol_date'),
        schema='economic_data'
    )
    
    # Create indexes for economic_data
    op.create_index('idx_econ_symbol_date', 'economic_data', ['symbol', 'date'], schema='economic_data')
    op.create_index('idx_econ_date', 'economic_data', ['date'], schema='economic_data')
    
    # Create economic_releases table
    op.create_table(
        'economic_releases',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('symbol', sa.String(50), nullable=False),
        sa.Column('release_name', sa.String(200), nullable=False),
        sa.Column('release_datetime', sa.DateTime(timezone=True), nullable=False),
        sa.Column('period', sa.String(50), nullable=False),
        sa.Column('actual', sa.Numeric(20, 8), nullable=True),
        sa.Column('forecast', sa.Numeric(20, 8), nullable=True),
        sa.Column('previous', sa.Numeric(20, 8), nullable=True),
        sa.Column('revised_previous', sa.Numeric(20, 8), nullable=True),
        sa.Column('impact_level', sa.String(20), nullable=False),
        sa.Column('surprise_magnitude', sa.Float(), nullable=True),
        sa.Column('forecast_count', sa.Integer(), nullable=True),
        sa.Column('forecast_std_dev', sa.Numeric(20, 8), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema='economic_data'
    )
    
    # Create indexes for economic_releases
    op.create_index('idx_release_datetime', 'economic_releases', ['release_datetime'], schema='economic_data')
    op.create_index('idx_release_symbol', 'economic_releases', ['symbol'], schema='economic_data')
    op.create_index('idx_release_impact', 'economic_releases', ['impact_level'], schema='economic_data')
    
    # Create economic_forecasts table
    op.create_table(
        'economic_forecasts',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('symbol', sa.String(50), nullable=False),
        sa.Column('target_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('forecast_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('forecast_value', sa.Numeric(20, 8), nullable=False),
        sa.Column('forecast_low', sa.Numeric(20, 8), nullable=True),
        sa.Column('forecast_high', sa.Numeric(20, 8), nullable=True),
        sa.Column('source', sa.String(100), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'target_date', 'source', 'forecast_date', 
                          name='uq_forecast_symbol_target_source_date'),
        schema='economic_data'
    )
    
    # Create indexes for economic_forecasts
    op.create_index('idx_forecast_symbol_target', 'economic_forecasts', ['symbol', 'target_date'], schema='economic_data')
    op.create_index('idx_forecast_date', 'economic_forecasts', ['forecast_date'], schema='economic_data')


def downgrade() -> None:
    # Drop tables
    op.drop_table('economic_forecasts', schema='economic_data')
    op.drop_table('economic_releases', schema='economic_data')
    op.drop_table('economic_data', schema='economic_data')
    
    # Drop schema
    op.execute("DROP SCHEMA IF EXISTS economic_data CASCADE")