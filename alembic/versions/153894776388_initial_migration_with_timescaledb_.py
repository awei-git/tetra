"""Initial migration with TimescaleDB tables

Revision ID: 153894776388
Revises: 
Create Date: 2025-07-30 22:13:29.270125

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '153894776388'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('signals',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('symbol', sa.String(length=20), nullable=False),
    sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
    sa.Column('signal_type', sa.Enum('BUY', 'SELL', 'HOLD', 'ENTER_LONG', 'EXIT_LONG', 'ENTER_SHORT', 'EXIT_SHORT', 'STOP_LOSS', 'TAKE_PROFIT', name='signaltype'), nullable=False),
    sa.Column('strength', sa.Float(), nullable=False),
    sa.Column('strategy_name', sa.String(length=100), nullable=False),
    sa.Column('timeframe', sa.String(length=10), nullable=False),
    sa.Column('current_price', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('target_price', sa.Numeric(precision=20, scale=8), nullable=True),
    sa.Column('stop_price', sa.Numeric(precision=20, scale=8), nullable=True),
    sa.Column('indicators_used', sa.JSON(), nullable=True),
    sa.Column('reasoning', sa.Text(), nullable=True),
    sa.Column('risk_reward_ratio', sa.Float(), nullable=True),
    sa.Column('position_size_suggestion', sa.Float(), nullable=True),
    sa.Column('backtest_id', sa.String(length=100), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    schema='derived'
    )
    op.create_index('idx_signal_strategy', 'signals', ['strategy_name'], unique=False, schema='derived')
    op.create_index('idx_signal_symbol_timestamp', 'signals', ['symbol', 'timestamp'], unique=False, schema='derived')
    op.create_index('idx_signal_type', 'signals', ['signal_type'], unique=False, schema='derived')
    op.create_table('technical_indicators',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('symbol', sa.String(length=20), nullable=False),
    sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
    sa.Column('indicator_type', sa.Enum('SMA', 'EMA', 'MACD', 'RSI', 'STOCH', 'BB', 'ATR', 'OBV', 'VWAP', 'CUSTOM', name='indicatortype'), nullable=False),
    sa.Column('timeframe', sa.String(length=10), nullable=False),
    sa.Column('parameters', sa.JSON(), nullable=False),
    sa.Column('value', sa.Numeric(precision=20, scale=8), nullable=True),
    sa.Column('values', sa.JSON(), nullable=True),
    sa.Column('calculation_time', sa.Float(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    schema='derived'
    )
    op.create_index('idx_indicator_symbol_timestamp', 'technical_indicators', ['symbol', 'timestamp'], unique=False, schema='derived')
    op.create_index('idx_indicator_type', 'technical_indicators', ['indicator_type'], unique=False, schema='derived')
    op.create_table('events',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('event_type', sa.Enum('ECONOMIC_RELEASE', 'CENTRAL_BANK', 'EARNINGS', 'DIVIDEND', 'SPLIT', 'MERGER', 'ELECTION', 'POLICY_CHANGE', 'CONFLICT', 'TRADE_DEAL', 'HALT', 'CIRCUIT_BREAKER', name='eventtype'), nullable=False),
    sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
    sa.Column('title', sa.String(length=500), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('impact_level', sa.String(length=20), nullable=False),
    sa.Column('affected_symbols', sa.JSON(), nullable=True),
    sa.Column('affected_sectors', sa.JSON(), nullable=True),
    sa.Column('source', sa.String(length=50), nullable=False),
    sa.Column('source_url', sa.String(length=1000), nullable=True),
    sa.Column('processed', sa.Boolean(), nullable=True),
    sa.Column('processing_notes', sa.Text(), nullable=True),
    sa.Column('event_data', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    schema='events'
    )
    # Skip indexing JSON columns - PostgreSQL can't create btree indexes on JSON
    # op.create_index('idx_event_affected_symbols', 'events', ['affected_symbols'], unique=False, schema='events')
    op.create_index('idx_event_timestamp', 'events', ['timestamp'], unique=False, schema='events')
    op.create_index('idx_event_type', 'events', ['event_type'], unique=False, schema='events')
    op.create_table('ohlcv',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('symbol', sa.String(length=20), nullable=False),
    sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
    sa.Column('open', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('high', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('low', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('close', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('volume', sa.Integer(), nullable=False),
    sa.Column('vwap', sa.Numeric(precision=20, scale=8), nullable=True),
    sa.Column('trades_count', sa.Integer(), nullable=True),
    sa.Column('timeframe', sa.String(length=10), nullable=False),
    sa.Column('source', sa.String(length=50), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('symbol', 'timestamp', 'timeframe', name='uq_ohlcv_symbol_time'),
    schema='market_data'
    )
    op.create_index('idx_ohlcv_symbol_timestamp', 'ohlcv', ['symbol', 'timestamp'], unique=False, schema='market_data')
    op.create_index('idx_ohlcv_timestamp', 'ohlcv', ['timestamp'], unique=False, schema='market_data')
    op.create_table('quotes',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('symbol', sa.String(length=20), nullable=False),
    sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
    sa.Column('bid_price', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('bid_size', sa.Integer(), nullable=False),
    sa.Column('ask_price', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('ask_size', sa.Integer(), nullable=False),
    sa.Column('bid_exchange', sa.String(length=10), nullable=True),
    sa.Column('ask_exchange', sa.String(length=10), nullable=True),
    sa.Column('source', sa.String(length=50), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    schema='market_data'
    )
    op.create_index('idx_quote_symbol_timestamp', 'quotes', ['symbol', 'timestamp'], unique=False, schema='market_data')
    op.create_index('idx_quote_timestamp', 'quotes', ['timestamp'], unique=False, schema='market_data')
    op.create_table('ticks',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('symbol', sa.String(length=20), nullable=False),
    sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
    sa.Column('price', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('size', sa.Integer(), nullable=False),
    sa.Column('conditions', sa.JSON(), nullable=True),
    sa.Column('exchange', sa.String(length=10), nullable=True),
    sa.Column('source', sa.String(length=50), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    schema='market_data'
    )
    op.create_index('idx_tick_symbol_timestamp', 'ticks', ['symbol', 'timestamp'], unique=False, schema='market_data')
    op.create_index('idx_tick_timestamp', 'ticks', ['timestamp'], unique=False, schema='market_data')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('idx_tick_timestamp', table_name='ticks', schema='market_data')
    op.drop_index('idx_tick_symbol_timestamp', table_name='ticks', schema='market_data')
    op.drop_table('ticks', schema='market_data')
    op.drop_index('idx_quote_timestamp', table_name='quotes', schema='market_data')
    op.drop_index('idx_quote_symbol_timestamp', table_name='quotes', schema='market_data')
    op.drop_table('quotes', schema='market_data')
    op.drop_index('idx_ohlcv_timestamp', table_name='ohlcv', schema='market_data')
    op.drop_index('idx_ohlcv_symbol_timestamp', table_name='ohlcv', schema='market_data')
    op.drop_table('ohlcv', schema='market_data')
    op.drop_index('idx_event_type', table_name='events', schema='events')
    op.drop_index('idx_event_timestamp', table_name='events', schema='events')
    op.drop_index('idx_event_affected_symbols', table_name='events', schema='events')
    op.drop_table('events', schema='events')
    op.drop_index('idx_indicator_type', table_name='technical_indicators', schema='derived')
    op.drop_index('idx_indicator_symbol_timestamp', table_name='technical_indicators', schema='derived')
    op.drop_table('technical_indicators', schema='derived')
    op.drop_index('idx_signal_type', table_name='signals', schema='derived')
    op.drop_index('idx_signal_symbol_timestamp', table_name='signals', schema='derived')
    op.drop_index('idx_signal_strategy', table_name='signals', schema='derived')
    op.drop_table('signals', schema='derived')
    # ### end Alembic commands ###