"""Create simulation schema and tables.

Run once to set up the simulation.* tables in PostgreSQL.

Usage:
  python scripts/migrate_simulation.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from src.db.session import engine

MIGRATION_SQL = """
-- Create simulation schema
CREATE SCHEMA IF NOT EXISTS simulation;

-- Regime detection results
CREATE TABLE IF NOT EXISTS simulation.regimes (
    as_of           DATE PRIMARY KEY,
    n_states        INTEGER NOT NULL,
    current_regime  VARCHAR(32) NOT NULL,
    current_probs   JSONB,
    transition_matrix JSONB,
    regime_states   JSONB,
    regime_forecast_5d JSONB,
    log_likelihood  NUMERIC(20, 4),
    n_observations  INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Covariance estimates
CREATE TABLE IF NOT EXISTS simulation.covariance (
    as_of               DATE NOT NULL,
    method              VARCHAR(64) NOT NULL,
    symbols             VARCHAR(32)[],
    vols_ann            JSONB,
    correlation_matrix  JSONB,
    n_observations      INTEGER,
    effective_observations NUMERIC(12, 4),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (as_of, method)
);

-- Portfolio risk metrics
CREATE TABLE IF NOT EXISTS simulation.risk (
    as_of               DATE NOT NULL,
    method              VARCHAR(32) NOT NULL,
    total_vol_ann       NUMERIC(12, 6),
    var_95_1d           NUMERIC(20, 4),
    var_99_1d           NUMERIC(20, 4),
    cvar_95_1d          NUMERIC(20, 4),
    cvar_99_1d          NUMERIC(20, 4),
    expected_max_drawdown NUMERIC(12, 6),
    hhi                 NUMERIC(12, 6),
    effective_n         NUMERIC(12, 4),
    marginal_risk       JSONB,
    component_risk      JSONB,
    risk_budget_breaches JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (as_of, method)
);

-- Scenario stress test results
CREATE TABLE IF NOT EXISTS simulation.scenarios (
    as_of               DATE NOT NULL,
    scenario_name       VARCHAR(128) NOT NULL,
    description         TEXT,
    portfolio_pnl       NUMERIC(20, 4),
    portfolio_pnl_pct   NUMERIC(12, 6),
    var_95_under_stress NUMERIC(20, 4),
    worst_position      VARCHAR(32),
    worst_position_pnl  NUMERIC(20, 4),
    target_moves        JSONB,
    position_pnls       JSONB,
    summary_stats       JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (as_of, scenario_name)
);

-- Indexes for time-series queries
CREATE INDEX IF NOT EXISTS idx_simulation_regimes_regime
    ON simulation.regimes (current_regime);
CREATE INDEX IF NOT EXISTS idx_simulation_risk_vol
    ON simulation.risk (as_of, total_vol_ann);
CREATE INDEX IF NOT EXISTS idx_simulation_scenarios_pnl
    ON simulation.scenarios (as_of, portfolio_pnl_pct);
"""


async def migrate() -> None:
    print("Creating simulation schema and tables...")
    async with engine.begin() as conn:
        for statement in MIGRATION_SQL.split(";"):
            statement = statement.strip()
            if statement:
                await conn.execute(text(statement))
    print("Migration complete.")


if __name__ == "__main__":
    asyncio.run(migrate())
