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


__all__ = [
    "metadata",
    "market_assets",
    "market_ohlcv",
    "event_events",
    "economic_series",
    "economic_values",
    "news_articles",
    "gpt_recommendations",
    "gpt_recommendation_challenges",
    "gpt_factor_reviews",
    "factors_daily",
    "factors_runs",
]
