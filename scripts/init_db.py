"""Create database schemas and tables."""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import settings
from src.db.schema import metadata

SCHEMAS = ("market", "event", "economic", "news", "gpt", "factors", "fundamentals")
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


if __name__ == "__main__":
    main()
