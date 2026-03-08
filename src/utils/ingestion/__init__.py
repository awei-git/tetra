"""Ingestion utilities for external data providers."""

from src.utils.ingestion.economic import ingest_economic_data
from src.utils.ingestion.events import ingest_event_data
from src.utils.ingestion.market import ingest_market_data
from src.utils.ingestion.polymarket import ingest_polymarket_data
from src.utils.ingestion.news import ingest_news_data
from src.utils.ingestion.types import IngestionSummary

__all__ = [
    "IngestionSummary",
    "ingest_economic_data",
    "ingest_event_data",
    "ingest_market_data",
    "ingest_polymarket_data",
    "ingest_news_data",
]
