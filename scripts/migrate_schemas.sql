-- Tetra schema migration: add narrative, signals, portfolio, tracker, report, network schemas
-- Run with: docker exec tetra-db psql -U tetra_user -d tetra -f /path/to/migrate_schemas.sql

-- ============================================================
-- narrative: LLM-extracted narrative state from news
-- ============================================================
CREATE SCHEMA IF NOT EXISTS narrative;

CREATE TABLE IF NOT EXISTS narrative.daily_state (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    scope TEXT NOT NULL,  -- symbol, sector, or 'market'
    scope_value TEXT NOT NULL,  -- e.g. 'AAPL', 'tech', 'market'
    dominant_narrative TEXT NOT NULL,
    narrative_shift FLOAT,  -- -1 to +1
    shift_magnitude FLOAT,  -- 0 to 1
    counter_narrative TEXT,
    novelty FLOAT,  -- 0 to 1, how new this narrative is
    historical_parallels JSONB,
    expected_impact JSONB,
    raw_analysis TEXT,  -- full LLM output
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (date, scope, scope_value)
);

-- ============================================================
-- signals: unified and cross-asset signals
-- ============================================================
CREATE SCHEMA IF NOT EXISTS signals;

CREATE TABLE IF NOT EXISTS signals.information_flow (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    source_market TEXT NOT NULL,  -- 'polymarket', 'rates', 'equities'
    target_market TEXT NOT NULL,
    gap_signal FLOAT,  -- how much target hasn't caught up
    gap_z_score FLOAT,
    context TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (date, source_market, target_market)
);

CREATE TABLE IF NOT EXISTS signals.polymarket_implied (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    market_id TEXT NOT NULL,
    question TEXT,
    implied_view TEXT,  -- LLM interpretation
    affected_assets JSONB,  -- [{symbol, direction, magnitude}]
    current_gap FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (date, market_id)
);

CREATE TABLE IF NOT EXISTS signals.informed_trading (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    signal_type TEXT NOT NULL,  -- 'insider_cluster', 'analyst_shift', 'pattern_break'
    strength FLOAT,
    historical_hit_rate FLOAT,
    context TEXT,
    raw_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (symbol, date, signal_type)
);

CREATE TABLE IF NOT EXISTS signals.unified (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    signal_score FLOAT NOT NULL,  -- -1 to +1
    signal_direction TEXT NOT NULL,  -- strong_buy/buy/neutral/sell/strong_sell
    confidence FLOAT,
    components JSONB,  -- {narrative: x, info_flow: y, earnings: z, informed: w, debate: v}
    conflicts JSONB,
    key_drivers JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (symbol, date)
);

-- ============================================================
-- network: company relationships and earnings cascade
-- ============================================================
CREATE SCHEMA IF NOT EXISTS network;

CREATE TABLE IF NOT EXISTS network.company_graph (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    related_symbol TEXT NOT NULL,
    relation_type TEXT NOT NULL,  -- 'supplier', 'customer', 'peer', 'competitor'
    strength FLOAT,  -- 0 to 1
    notes TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (symbol, related_symbol, relation_type)
);

CREATE TABLE IF NOT EXISTS network.earnings_cascade (
    id SERIAL PRIMARY KEY,
    source_symbol TEXT NOT NULL,
    target_symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,  -- 'guidance_positive', 'demand_confirmation', etc
    magnitude FLOAT,
    confidence FLOAT,
    date DATE NOT NULL,
    context TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- portfolio: position tracking and analytics
-- ============================================================
CREATE SCHEMA IF NOT EXISTS portfolio;

CREATE TABLE IF NOT EXISTS portfolio.positions (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL UNIQUE,
    shares FLOAT NOT NULL,
    avg_cost FLOAT NOT NULL,
    entry_date DATE,
    current_price FLOAT,
    market_value FLOAT,
    unrealized_pnl FLOAT,
    weight FLOAT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS portfolio.cash (
    id SERIAL PRIMARY KEY,
    amount FLOAT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS portfolio.snapshots (
    date DATE PRIMARY KEY,
    total_value FLOAT,
    cash FLOAT,
    invested FLOAT,
    daily_return FLOAT,
    cumulative_return FLOAT,
    positions JSONB
);

CREATE TABLE IF NOT EXISTS portfolio.analytics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    metric_name TEXT NOT NULL,
    value FLOAT NOT NULL,
    window_days INTEGER,
    UNIQUE (date, metric_name, window_days)
);

-- ============================================================
-- tracker: recommendation tracking
-- ============================================================
CREATE SCHEMA IF NOT EXISTS tracker;

CREATE TABLE IF NOT EXISTS tracker.recommendations (
    id SERIAL PRIMARY KEY,
    created_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,  -- 'long' or 'short'
    entry_price FLOAT NOT NULL,
    target_price FLOAT,
    stop_loss FLOAT,
    status TEXT NOT NULL DEFAULT 'open',  -- open/hit_target/hit_stop/expired/closed
    closed_date DATE,
    closed_price FLOAT,
    realized_pnl FLOAT,
    max_favorable FLOAT,
    max_adverse FLOAT,
    confidence TEXT,  -- high/medium/low
    time_horizon TEXT,  -- intraday/swing/position
    method TEXT,  -- narrative/info_flow/earnings_network/debate_consensus/debate_contrarian
    thesis TEXT,
    risk_factors TEXT,
    supporting_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tracker.daily_marks (
    id SERIAL PRIMARY KEY,
    rec_id INTEGER REFERENCES tracker.recommendations(id),
    date DATE NOT NULL,
    price FLOAT NOT NULL,
    unrealized_pnl FLOAT,
    UNIQUE (rec_id, date)
);

-- ============================================================
-- report: generated reports
-- ============================================================
CREATE SCHEMA IF NOT EXISTS report;

CREATE TABLE IF NOT EXISTS report.daily (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    pdf_path TEXT,
    summary TEXT,
    regime TEXT,
    portfolio_pnl FLOAT,
    new_recommendations INTEGER,
    active_recommendations INTEGER,
    sections JSONB,
    llm_stats JSONB,
    duration_seconds FLOAT
);

-- ============================================================
-- Phase 2: insider trades + analyst recommendations (raw data)
-- ============================================================

CREATE TABLE IF NOT EXISTS event.insider_trades (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    filing_date DATE NOT NULL,
    transaction_date DATE,
    insider_name TEXT,
    insider_title TEXT,
    transaction_type TEXT,  -- 'P-Purchase', 'S-Sale', etc.
    shares FLOAT,
    price FLOAT,
    value FLOAT,
    shares_after FLOAT,
    source TEXT NOT NULL DEFAULT 'finnhub',
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (symbol, filing_date, insider_name, transaction_type, shares)
);

CREATE INDEX IF NOT EXISTS idx_insider_trades_symbol_date
    ON event.insider_trades (symbol, filing_date DESC);

CREATE TABLE IF NOT EXISTS event.analyst_recommendations (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    period DATE NOT NULL,
    strong_buy INTEGER DEFAULT 0,
    buy INTEGER DEFAULT 0,
    hold INTEGER DEFAULT 0,
    sell INTEGER DEFAULT 0,
    strong_sell INTEGER DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'finnhub',
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (symbol, period, source)
);

CREATE INDEX IF NOT EXISTS idx_analyst_recs_symbol_period
    ON event.analyst_recommendations (symbol, period DESC);

CREATE TABLE IF NOT EXISTS network.analyst_coverage (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    peer_symbol TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'finnhub',
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (symbol, peer_symbol, source)
);

CREATE TABLE IF NOT EXISTS network.supply_chain (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    related_symbol TEXT NOT NULL,
    relation_type TEXT NOT NULL,  -- 'customer', 'supplier'
    revenue_pct FLOAT,
    source TEXT NOT NULL DEFAULT 'finnhub',
    payload JSONB,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (symbol, related_symbol, relation_type, source)
);
