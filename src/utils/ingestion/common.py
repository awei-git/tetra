"""Shared helpers for ingestion tasks."""

from __future__ import annotations

from datetime import date
from typing import Dict, Iterable, Iterator, List, Optional


def chunk(items: Iterable[Dict], size: int = 500) -> Iterator[List[Dict]]:
    batch: List[Dict] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None
