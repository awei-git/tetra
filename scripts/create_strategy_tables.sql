-- Create strategy backtest tables

-- Create strategies schema
CREATE SCHEMA IF NOT EXISTS strategies;

-- Create backtest_results table
CREATE TABLE IF NOT EXISTS strategies.backtest_results (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR NOT NULL,
    run_date TIMESTAMP NOT NULL,
    backtest_start_date DATE NOT NULL,
    backtest_end_date DATE NOT NULL,
    universe VARCHAR NOT NULL,
    initial_capital FLOAT NOT NULL,
    final_value FLOAT,
    total_return FLOAT,
    annualized_return FLOAT,
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    volatility FLOAT,
    win_rate FLOAT,
    total_trades INTEGER,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create index
CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy_date 
ON strategies.backtest_results(strategy_name, run_date);

-- Create strategy_rankings table
CREATE TABLE IF NOT EXISTS strategies.strategy_rankings (
    id SERIAL PRIMARY KEY,
    run_date TIMESTAMP NOT NULL,
    strategy_name VARCHAR NOT NULL,
    rank_by_sharpe INTEGER,
    rank_by_return INTEGER,
    rank_by_consistency INTEGER,
    composite_score FLOAT,
    category VARCHAR,
    overall_rank INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create index
CREATE INDEX IF NOT EXISTS idx_strategy_rankings_run_date 
ON strategies.strategy_rankings(run_date);

-- Create backtest_summary table
CREATE TABLE IF NOT EXISTS strategies.backtest_summary (
    id SERIAL PRIMARY KEY,
    run_date TIMESTAMP NOT NULL,
    total_strategies INTEGER NOT NULL,
    successful_strategies INTEGER NOT NULL,
    avg_return FLOAT,
    avg_sharpe FLOAT,
    avg_max_drawdown FLOAT,
    best_return FLOAT,
    worst_return FLOAT,
    best_sharpe FLOAT,
    execution_time FLOAT,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create strategy_metadata table
CREATE TABLE IF NOT EXISTS strategies.strategy_metadata (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR NOT NULL UNIQUE,
    category VARCHAR,
    description TEXT,
    last_backtest_date TIMESTAMP,
    last_sharpe_ratio FLOAT,
    last_total_return FLOAT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP
);

-- Create equity_curves table
CREATE TABLE IF NOT EXISTS strategies.equity_curves (
    id SERIAL PRIMARY KEY,
    backtest_id INTEGER NOT NULL,
    strategy_name VARCHAR NOT NULL,
    dates JSONB NOT NULL,
    values JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create index
CREATE INDEX IF NOT EXISTS idx_equity_curves_backtest_id 
ON strategies.equity_curves(backtest_id);