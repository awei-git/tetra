"""Shared types for ingestion helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class IngestionSummary:
    records: int
    details: Dict[str, Any] = field(default_factory=dict)
