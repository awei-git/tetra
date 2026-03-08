"""Create database schemas and tables."""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import settings
from src.db.schema import metadata

SCHEMAS = (
    "market",
    "event",
    "economic",
    "news",
    "gpt",
    "factors",
    "fundamentals",
    "polymarket",
    "inference",
)
UNIQUE_INDEXES = (
    (
        "event.events",
        "event_events_source_external_id_event_time_key",
        "source, external_id, event_time",
    ),
    (
        "news.articles",
        "news_articles_source_external_id_published_at_key",
        "source, external_id, published_at",
    ),
    (
        "inference.signal_leaderboard",
        "inference_signal_leaderboard_factor_horizon_asof_key",
        "factor, horizon_days, as_of",
    ),
    (
        "inference.event_study",
        "inference_event_study_event_type_window_asof_key",
        "event_type, window_days, as_of",
    ),
)

PRIMARY_KEY_FIXES = (
    (
        "inference.signal_leaderboard",
        "signal_leaderboard_pkey",
        "factor, horizon_days, as_of",
    ),
    (
        "inference.event_study",
        "event_study_pkey",
        "event_type, window_days, as_of",
    ),
)


def main() -> None:
    engine = create_engine(settings.sync_database_url, future=True)
    with engine.begin() as conn:
        for schema in SCHEMAS:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
    metadata.create_all(engine)
    with engine.begin() as conn:
        for table_name, index_name, columns in UNIQUE_INDEXES:
            conn.execute(
                text(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} "
                    f"ON {table_name} ({columns})"
                )
            )
        for table_name, pk_name, columns in PRIMARY_KEY_FIXES:
            conn.execute(text(f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {pk_name}"))
            conn.execute(text(f"ALTER TABLE {table_name} ADD CONSTRAINT {pk_name} PRIMARY KEY ({columns})"))


if __name__ == "__main__":
    main()
