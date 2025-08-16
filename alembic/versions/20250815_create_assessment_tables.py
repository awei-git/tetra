"""Create assessment tables for storing pipeline results

Revision ID: 20250815_assessment
Revises: 
Create Date: 2025-08-15

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '20250815_assessment'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create assessment schema and tables."""
    
    # Create schema if not exists
    op.execute("CREATE SCHEMA IF NOT EXISTS assessment")
    
    # Create backtest_results table
    op.create_table(
        'backtest_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_date', sa.TIMESTAMP(), nullable=False),
        sa.Column('strategy_name', sa.String(100), nullable=False),
        sa.Column('strategy_category', sa.String(50), nullable=True),
        sa.Column('scenario_name', sa.String(100), nullable=False),
        sa.Column('scenario_type', sa.String(50), nullable=True),
        sa.Column('symbol', sa.String(50), nullable=False),
        sa.Column('total_return', sa.Numeric(10, 4), nullable=True),
        sa.Column('annual_return', sa.Numeric(10, 4), nullable=True),
        sa.Column('sharpe_ratio', sa.Numeric(10, 4), nullable=True),
        sa.Column('sortino_ratio', sa.Numeric(10, 4), nullable=True),
        sa.Column('max_drawdown', sa.Numeric(10, 4), nullable=True),
        sa.Column('calmar_ratio', sa.Numeric(10, 4), nullable=True),
        sa.Column('win_rate', sa.Numeric(10, 4), nullable=True),
        sa.Column('profit_factor', sa.Numeric(10, 4), nullable=True),
        sa.Column('volatility', sa.Numeric(10, 4), nullable=True),
        sa.Column('score', sa.Numeric(10, 4), nullable=True),
        sa.Column('total_trades', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema='assessment'
    )
    
    # Create indexes for common queries
    op.create_index(
        'idx_backtest_run_date',
        'backtest_results',
        ['run_date'],
        schema='assessment'
    )
    op.create_index(
        'idx_backtest_strategy',
        'backtest_results',
        ['strategy_name'],
        schema='assessment'
    )
    op.create_index(
        'idx_backtest_scenario',
        'backtest_results',
        ['scenario_name'],
        schema='assessment'
    )
    op.create_index(
        'idx_backtest_symbol',
        'backtest_results',
        ['symbol'],
        schema='assessment'
    )
    op.create_index(
        'idx_backtest_score',
        'backtest_results',
        ['score'],
        schema='assessment'
    )
    
    # Create strategy_rankings table
    op.create_table(
        'strategy_rankings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_date', sa.TIMESTAMP(), nullable=False),
        sa.Column('strategy_name', sa.String(100), nullable=False),
        sa.Column('category', sa.String(50), nullable=True),
        sa.Column('overall_rank', sa.Integer(), nullable=True),
        sa.Column('category_rank', sa.Integer(), nullable=True),
        sa.Column('avg_score', sa.Numeric(10, 4), nullable=True),
        sa.Column('avg_return', sa.Numeric(10, 4), nullable=True),
        sa.Column('avg_sharpe', sa.Numeric(10, 4), nullable=True),
        sa.Column('avg_drawdown', sa.Numeric(10, 4), nullable=True),
        sa.Column('num_tests', sa.Integer(), nullable=True),
        sa.Column('best_scenario', sa.String(100), nullable=True),
        sa.Column('worst_scenario', sa.String(100), nullable=True),
        sa.Column('best_symbol', sa.String(50), nullable=True),
        sa.Column('worst_symbol', sa.String(50), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema='assessment'
    )
    
    # Create indexes for rankings
    op.create_index(
        'idx_ranking_run_date',
        'strategy_rankings',
        ['run_date'],
        schema='assessment'
    )
    op.create_index(
        'idx_ranking_overall',
        'strategy_rankings',
        ['overall_rank'],
        schema='assessment'
    )
    op.create_index(
        'idx_ranking_category',
        'strategy_rankings',
        ['category', 'category_rank'],
        schema='assessment'
    )
    
    # Create scenario_performance table for quick lookups
    op.create_table(
        'scenario_performance',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_date', sa.TIMESTAMP(), nullable=False),
        sa.Column('scenario_name', sa.String(100), nullable=False),
        sa.Column('scenario_type', sa.String(50), nullable=True),
        sa.Column('top_strategy', sa.String(100), nullable=True),
        sa.Column('top_symbol', sa.String(50), nullable=True),
        sa.Column('top_return', sa.Numeric(10, 4), nullable=True),
        sa.Column('top_score', sa.Numeric(10, 4), nullable=True),
        sa.Column('avg_return', sa.Numeric(10, 4), nullable=True),
        sa.Column('strategies_tested', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema='assessment'
    )
    
    op.create_index(
        'idx_scenario_perf_run_date',
        'scenario_performance',
        ['run_date'],
        schema='assessment'
    )
    op.create_index(
        'idx_scenario_perf_name',
        'scenario_performance',
        ['scenario_name'],
        schema='assessment'
    )
    
    # Create pipeline_runs table to track pipeline executions
    op.create_table(
        'pipeline_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('pipeline_name', sa.String(50), nullable=False),
        sa.Column('start_time', sa.TIMESTAMP(), nullable=False),
        sa.Column('end_time', sa.TIMESTAMP(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False),  # running, success, failed
        sa.Column('records_processed', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('execution_time_seconds', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema='assessment'
    )
    
    op.create_index(
        'idx_pipeline_runs_name',
        'pipeline_runs',
        ['pipeline_name', 'start_time'],
        schema='assessment'
    )
    op.create_index(
        'idx_pipeline_runs_status',
        'pipeline_runs',
        ['status'],
        schema='assessment'
    )


def downgrade():
    """Drop assessment schema and tables."""
    op.drop_table('pipeline_runs', schema='assessment')
    op.drop_table('scenario_performance', schema='assessment')
    op.drop_table('strategy_rankings', schema='assessment')
    op.drop_table('backtest_results', schema='assessment')
    op.execute("DROP SCHEMA IF EXISTS assessment CASCADE")