"""SQLAlchemy table definitions matching migration schema."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

metadata = sa.MetaData()

market_assets = sa.Table(
    "assets",
    metadata,
    sa.Column("symbol", sa.String(32), primary_key=True),
    sa.Column("asset_type", sa.String(32), nullable=False),
    sa.Column("name", sa.String(255)),
    sa.Column("exchange", sa.String(64)),
    sa.Column("currency", sa.String(16)),
    sa.Column("sector", sa.String(128)),
    sa.Column("industry", sa.String(128)),
    sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    sa.Column("listed_at", sa.Date()),
    sa.Column("delisted_at", sa.Date()),
    sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="market",
)

market_ohlcv = sa.Table(
    "ohlcv",
    metadata,
    sa.Column("symbol", sa.String(32), primary_key=True),
    sa.Column("timestamp", sa.DateTime(timezone=True), primary_key=True),
    sa.Column("open", sa.Numeric(20, 8)),
    sa.Column("high", sa.Numeric(20, 8)),
    sa.Column("low", sa.Numeric(20, 8)),
    sa.Column("close", sa.Numeric(20, 8), nullable=False),
    sa.Column("volume", sa.BigInteger()),
    sa.Column("vwap", sa.Numeric(20, 8)),
    sa.Column("turnover", sa.Numeric(20, 8)),
    sa.Column("source", sa.String(64)),
    sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="market",
)


polymarket_markets = sa.Table(
    "markets",
    metadata,
    sa.Column("market_id", sa.String(64), primary_key=True),
    sa.Column("slug", sa.String(255)),
    sa.Column("question", sa.Text()),
    sa.Column("category", sa.String(128)),
    sa.Column("description", sa.Text()),
    sa.Column("active", sa.Boolean()),
    sa.Column("closed", sa.Boolean()),
    sa.Column("archived", sa.Boolean()),
    sa.Column("end_time", sa.DateTime(timezone=True)),
    sa.Column("created_time", sa.DateTime(timezone=True)),
    sa.Column("volume", sa.Numeric(20, 8)),
    sa.Column("liquidity", sa.Numeric(20, 8)),
    sa.Column("best_bid", sa.Numeric(20, 8)),
    sa.Column("best_ask", sa.Numeric(20, 8)),
    sa.Column("condition_id", sa.String(128)),
    sa.Column("clob_token_ids", postgresql.ARRAY(sa.String(128))),
    sa.Column("payload", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="polymarket",
)


polymarket_snapshots = sa.Table(
    "snapshots",
    metadata,
    sa.Column("market_id", sa.String(64), primary_key=True),
    sa.Column("snapshot_time", sa.DateTime(timezone=True), primary_key=True),
    sa.Column("active", sa.Boolean()),
    sa.Column("closed", sa.Boolean()),
    sa.Column("volume", sa.Numeric(20, 8)),
    sa.Column("liquidity", sa.Numeric(20, 8)),
    sa.Column("best_bid", sa.Numeric(20, 8)),
    sa.Column("best_ask", sa.Numeric(20, 8)),
    sa.Column("payload", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="polymarket",
)


inference_signal_leaderboard = sa.Table(
    "signal_leaderboard",
    metadata,
    sa.Column("factor", sa.String(128), primary_key=True),
    sa.Column("horizon_days", sa.Integer(), primary_key=True),
    sa.Column("as_of", sa.Date(), primary_key=True, nullable=False),
    sa.Column("start_date", sa.Date()),
    sa.Column("end_date", sa.Date()),
    sa.Column("avg_ic", sa.Numeric(8, 4)),
    sa.Column("median_ic", sa.Numeric(8, 4)),
    sa.Column("hit_rate", sa.Numeric(8, 4)),
    sa.Column("days", sa.Integer()),
    sa.Column("observations", sa.Integer()),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="inference",
)


inference_event_study = sa.Table(
    "event_study",
    metadata,
    sa.Column("event_type", sa.String(64), primary_key=True),
    sa.Column("window_days", sa.Integer(), primary_key=True),
    sa.Column("as_of", sa.Date(), primary_key=True, nullable=False),
    sa.Column("start_date", sa.Date()),
    sa.Column("end_date", sa.Date()),
    sa.Column("avg_return", sa.Numeric(12, 6)),
    sa.Column("median_return", sa.Numeric(12, 6)),
    sa.Column("observations", sa.Integer()),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="inference",
)


inference_polymarket_summary = sa.Table(
    "polymarket_summary",
    metadata,
    sa.Column("as_of", sa.Date(), primary_key=True),
    sa.Column("markets", sa.Integer()),
    sa.Column("closed_markets", sa.Integer()),
    sa.Column("resolved_proxy", sa.Integer()),
    sa.Column("avg_spread", sa.Numeric(12, 6)),
    sa.Column("avg_volume", sa.Numeric(20, 8)),
    sa.Column("avg_brier", sa.Numeric(12, 6)),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="inference",
)


inference_polymarket_bins = sa.Table(
    "polymarket_bins",
    metadata,
    sa.Column("as_of", sa.Date(), primary_key=True),
    sa.Column("bin_low", sa.Numeric(5, 2), primary_key=True),
    sa.Column("bin_high", sa.Numeric(5, 2), primary_key=True),
    sa.Column("count", sa.Integer()),
    sa.Column("avg_pred", sa.Numeric(12, 6)),
    sa.Column("proxy_accuracy", sa.Numeric(12, 6)),
    sa.Column("avg_brier", sa.Numeric(12, 6)),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="inference",
)


fundamentals_financials = sa.Table(
    "financials",
    metadata,
    sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
    sa.Column("symbol", sa.String(32), nullable=False),
    sa.Column("timeframe", sa.String(16)),
    sa.Column("fiscal_year", sa.Integer()),
    sa.Column("fiscal_period", sa.String(16)),
    sa.Column("period_end", sa.Date()),
    sa.Column("filing_date", sa.Date()),
    sa.Column("source", sa.String(32), nullable=False),
    sa.Column("payload", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    sa.UniqueConstraint(
        "symbol",
        "timeframe",
        "fiscal_year",
        "fiscal_period",
        "source",
        name="fund_financials_unique",
    ),
    schema="fundamentals",
)


fundamentals_shares = sa.Table(
    "shares",
    metadata,
    sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
    sa.Column("symbol", sa.String(32), nullable=False),
    sa.Column("as_of", sa.Date(), nullable=False),
    sa.Column("share_class_shares_outstanding", sa.Numeric(20, 4)),
    sa.Column("weighted_shares_outstanding", sa.Numeric(20, 4)),
    sa.Column("market_cap", sa.Numeric(20, 4)),
    sa.Column("source", sa.String(32), nullable=False),
    sa.Column("payload", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    sa.UniqueConstraint(
        "symbol",
        "as_of",
        "source",
        name="fundamentals_shares_symbol_as_of_source_key",
    ),
    schema="fundamentals",
)


event_events = sa.Table(
    "events",
    metadata,
    sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
    sa.Column("symbol", sa.String(32)),
    sa.Column("event_type", sa.String(64), nullable=False),
    sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("external_id", sa.String(512)),
    sa.Column("source", sa.String(64)),
    sa.Column("importance", sa.String(32)),
    sa.Column("payload", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    sa.UniqueConstraint(
        "source",
        "external_id",
        "event_time",
        name="event_events_source_external_id_event_time_key",
    ),
    schema="event",
)


economic_series = sa.Table(
    "series",
    metadata,
    sa.Column("series_id", sa.String(64), primary_key=True),
    sa.Column("name", sa.String(255), nullable=False),
    sa.Column("frequency", sa.String(32)),
    sa.Column("unit", sa.String(64)),
    sa.Column("seasonal_adjustment", sa.String(64)),
    sa.Column("region", sa.String(64)),
    sa.Column("data_source", sa.String(64)),
    sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="economic",
)


economic_values = sa.Table(
    "values",
    metadata,
    sa.Column("series_id", sa.String(64), primary_key=True),
    sa.Column("timestamp", sa.DateTime(timezone=True), primary_key=True),
    sa.Column("value", sa.Numeric(20, 8), nullable=False),
    sa.Column("revision", sa.Integer()),
    sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="economic",
)


news_articles = sa.Table(
    "articles",
    metadata,
    sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
    sa.Column("external_id", sa.String(512)),
    sa.Column("headline", sa.Text(), nullable=False),
    sa.Column("summary", sa.Text()),
    sa.Column("url", sa.Text()),
    sa.Column("source", sa.String(64)),
    sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("tickers", postgresql.ARRAY(sa.String(32))),
    sa.Column("sentiment", sa.Numeric(8, 4)),
    sa.Column("sentiment_confidence", sa.Numeric(8, 4)),
    sa.Column("embeddings", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("payload", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    sa.UniqueConstraint(
        "source",
        "external_id",
        "published_at",
        name="news_articles_source_external_id_published_at_key",
    ),
    schema="news",
)


gpt_recommendations = sa.Table(
    "recommendations",
    metadata,
    sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
    sa.Column("provider", sa.String(32), nullable=False),
    sa.Column("session", sa.String(16), nullable=False),
    sa.Column("run_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("payload", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("raw_text", sa.Text()),
    sa.Column("error", sa.Text()),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    sa.UniqueConstraint(
        "provider",
        "session",
        "run_time",
        name="gpt_recommendations_provider_session_run_time_key",
    ),
    schema="gpt",
)

gpt_recommendation_challenges = sa.Table(
    "recommendation_challenges",
    metadata,
    sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
    sa.Column("provider", sa.String(32), nullable=False),
    sa.Column("session", sa.String(16), nullable=False),
    sa.Column("run_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("source_provider", sa.String(32)),
    sa.Column("source_run_time", sa.DateTime(timezone=True)),
    sa.Column("source_payload", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("payload", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("raw_text", sa.Text()),
    sa.Column("error", sa.Text()),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    sa.UniqueConstraint(
        "provider",
        "session",
        "run_time",
        name="gpt_recommendation_challenges_provider_session_run_time_key",
    ),
    schema="gpt",
)

gpt_factor_reviews = sa.Table(
    "factor_reviews",
    metadata,
    sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
    sa.Column("provider", sa.String(32), nullable=False),
    sa.Column("session", sa.String(16), nullable=False),
    sa.Column("run_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("as_of", sa.Date(), nullable=False),
    sa.Column("payload", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("raw_text", sa.Text()),
    sa.Column("error", sa.Text()),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    sa.UniqueConstraint(
        "provider",
        "session",
        "run_time",
        name="gpt_factor_reviews_provider_session_run_time_key",
    ),
    schema="gpt",
)

gpt_recommendation_summaries = sa.Table(
    "recommendation_summaries",
    metadata,
    sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
    sa.Column("session", sa.String(16)),
    sa.Column("run_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("as_of", sa.Date()),
    sa.Column("provider", sa.String(32)),
    sa.Column("payload", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("raw_text", sa.Text()),
    sa.Column("error", sa.Text()),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    sa.UniqueConstraint(
        "session",
        "run_time",
        name="gpt_recommendation_summaries_session_run_time_key",
    ),
    schema="gpt",
)


factors_daily = sa.Table(
    "daily_factors",
    metadata,
    sa.Column("symbol", sa.String(32), primary_key=True),
    sa.Column("as_of", sa.Date(), primary_key=True),
    sa.Column("factor", sa.String(128), primary_key=True),
    sa.Column("value", sa.Numeric(20, 8)),
    sa.Column("source", sa.String(32)),
    sa.Column("window_days", sa.Integer()),
    sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="factors",
)

factors_runs = sa.Table(
    "factor_runs",
    metadata,
    sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
    sa.Column("as_of", sa.Date(), nullable=False),
    sa.Column("run_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("status", sa.String(16), nullable=False),
    sa.Column("summary", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("error", sa.Text()),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="factors",
)


# ---------------------------------------------------------------------------
# Simulation schema — regime detection, risk metrics, scenario analysis
# ---------------------------------------------------------------------------

simulation_regimes = sa.Table(
    "regimes",
    metadata,
    sa.Column("as_of", sa.Date(), primary_key=True),
    sa.Column("n_states", sa.Integer(), nullable=False),
    sa.Column("current_regime", sa.String(32), nullable=False),
    sa.Column("current_probs", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("transition_matrix", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("regime_states", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("regime_forecast_5d", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("log_likelihood", sa.Numeric(20, 4)),
    sa.Column("n_observations", sa.Integer()),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="simulation",
)

simulation_risk = sa.Table(
    "risk",
    metadata,
    sa.Column("as_of", sa.Date(), primary_key=True),
    sa.Column("method", sa.String(32), primary_key=True),
    sa.Column("total_vol_ann", sa.Numeric(12, 6)),
    sa.Column("var_95_1d", sa.Numeric(20, 4)),
    sa.Column("var_99_1d", sa.Numeric(20, 4)),
    sa.Column("cvar_95_1d", sa.Numeric(20, 4)),
    sa.Column("cvar_99_1d", sa.Numeric(20, 4)),
    sa.Column("expected_max_drawdown", sa.Numeric(12, 6)),
    sa.Column("hhi", sa.Numeric(12, 6)),
    sa.Column("effective_n", sa.Numeric(12, 4)),
    sa.Column("marginal_risk", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("component_risk", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("risk_budget_breaches", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="simulation",
)

simulation_scenarios = sa.Table(
    "scenarios",
    metadata,
    sa.Column("as_of", sa.Date(), primary_key=True),
    sa.Column("scenario_name", sa.String(128), primary_key=True),
    sa.Column("description", sa.Text()),
    sa.Column("portfolio_pnl", sa.Numeric(20, 4)),
    sa.Column("portfolio_pnl_pct", sa.Numeric(12, 6)),
    sa.Column("var_95_under_stress", sa.Numeric(20, 4)),
    sa.Column("worst_position", sa.String(32)),
    sa.Column("worst_position_pnl", sa.Numeric(20, 4)),
    sa.Column("target_moves", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("position_pnls", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("summary_stats", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="simulation",
)

simulation_covariance = sa.Table(
    "covariance",
    metadata,
    sa.Column("as_of", sa.Date(), primary_key=True),
    sa.Column("method", sa.String(64), primary_key=True),
    sa.Column("symbols", postgresql.ARRAY(sa.String(32))),
    sa.Column("vols_ann", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("correlation_matrix", postgresql.JSONB(astext_type=sa.Text())),
    sa.Column("n_observations", sa.Integer()),
    sa.Column("effective_observations", sa.Numeric(12, 4)),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    schema="simulation",
)


__all__ = [
    "metadata",
    "market_assets",
    "market_ohlcv",
    "polymarket_markets",
    "polymarket_snapshots",
    "inference_signal_leaderboard",
    "inference_event_study",
    "inference_polymarket_summary",
    "inference_polymarket_bins",
    "event_events",
    "economic_series",
    "economic_values",
    "news_articles",
    "gpt_recommendations",
    "gpt_recommendation_challenges",
    "gpt_factor_reviews",
    "gpt_recommendation_summaries",
    "factors_daily",
    "factors_runs",
    "simulation_regimes",
    "simulation_risk",
    "simulation_scenarios",
    "simulation_covariance",
]
