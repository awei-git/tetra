"""Create strategy backtest tables

Revision ID: create_strategy_tables
Revises: 
Create Date: 2025-01-06
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'create_strategy_tables'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create strategies schema if not exists
    op.execute("CREATE SCHEMA IF NOT EXISTS strategies")
    
    # Create backtest_results table
    op.create_table(
        'backtest_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('strategy_name', sa.String(), nullable=False),
        sa.Column('run_date', sa.DateTime(), nullable=False),
        sa.Column('backtest_start_date', sa.Date(), nullable=False),
        sa.Column('backtest_end_date', sa.Date(), nullable=False),
        sa.Column('universe', sa.String(), nullable=False),
        sa.Column('initial_capital', sa.Float(), nullable=False),
        sa.Column('final_value', sa.Float(), nullable=True),
        sa.Column('total_return', sa.Float(), nullable=True),
        sa.Column('annualized_return', sa.Float(), nullable=True),
        sa.Column('sharpe_ratio', sa.Float(), nullable=True),
        sa.Column('max_drawdown', sa.Float(), nullable=True),
        sa.Column('volatility', sa.Float(), nullable=True),
        sa.Column('win_rate', sa.Float(), nullable=True),
        sa.Column('total_trades', sa.Integer(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema='strategies'
    )
    
    # Create index on strategy_name and run_date
    op.create_index(
        'idx_backtest_results_strategy_date',
        'backtest_results',
        ['strategy_name', 'run_date'],
        schema='strategies'
    )
    
    # Create strategy_rankings table
    op.create_table(
        'strategy_rankings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_date', sa.DateTime(), nullable=False),
        sa.Column('strategy_name', sa.String(), nullable=False),
        sa.Column('rank_by_sharpe', sa.Integer(), nullable=True),
        sa.Column('rank_by_return', sa.Integer(), nullable=True),
        sa.Column('rank_by_consistency', sa.Integer(), nullable=True),
        sa.Column('composite_score', sa.Float(), nullable=True),
        sa.Column('category', sa.String(), nullable=True),
        sa.Column('overall_rank', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema='strategies'
    )
    
    # Create index on run_date
    op.create_index(
        'idx_strategy_rankings_run_date',
        'strategy_rankings',
        ['run_date'],
        schema='strategies'
    )
    
    # Create backtest_summary table
    op.create_table(
        'backtest_summary',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_date', sa.DateTime(), nullable=False),
        sa.Column('total_strategies', sa.Integer(), nullable=False),
        sa.Column('successful_strategies', sa.Integer(), nullable=False),
        sa.Column('avg_return', sa.Float(), nullable=True),
        sa.Column('avg_sharpe', sa.Float(), nullable=True),
        sa.Column('avg_max_drawdown', sa.Float(), nullable=True),
        sa.Column('best_return', sa.Float(), nullable=True),
        sa.Column('worst_return', sa.Float(), nullable=True),
        sa.Column('best_sharpe', sa.Float(), nullable=True),
        sa.Column('execution_time', sa.Float(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema='strategies'
    )
    
    # Create strategy_metadata table
    op.create_table(
        'strategy_metadata',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('strategy_name', sa.String(), nullable=False, unique=True),
        sa.Column('category', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('last_backtest_date', sa.DateTime(), nullable=True),
        sa.Column('last_sharpe_ratio', sa.Float(), nullable=True),
        sa.Column('last_total_return', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('strategy_name'),
        schema='strategies'
    )
    
    # Create equity_curves table for storing sampled equity curves
    op.create_table(
        'equity_curves',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('backtest_id', sa.Integer(), nullable=False),
        sa.Column('strategy_name', sa.String(), nullable=False),
        sa.Column('dates', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('values', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema='strategies'
    )
    
    # Create index on backtest_id
    op.create_index(
        'idx_equity_curves_backtest_id',
        'equity_curves',
        ['backtest_id'],
        schema='strategies'
    )


def downgrade():
    # Drop tables in reverse order
    op.drop_table('equity_curves', schema='strategies')
    op.drop_table('strategy_metadata', schema='strategies')
    op.drop_table('backtest_summary', schema='strategies')
    op.drop_table('strategy_rankings', schema='strategies')
    op.drop_table('backtest_results', schema='strategies')
    
    # Drop schema if empty
    op.execute("DROP SCHEMA IF EXISTS strategies CASCADE")