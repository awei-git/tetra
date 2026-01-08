"""Economic data ingestion helpers."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Sequence

import httpx
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.schema import economic_series, economic_values
from src.db.session import engine
from src.definitions.economic_indicators import INDICATORS
from src.utils.ingestion.clients import FREDClient
from src.utils.ingestion.common import chunk
from src.utils.ingestion.types import IngestionSummary

UTC = timezone.utc


async def ingest_economic_data(
    start: date,
    end: date,
    series_ids: Optional[Sequence[str]] = None,
) -> IngestionSummary:
    try:
        client = FREDClient()
    except RuntimeError as exc:
        print(f"Economic ingestion disabled: {exc}")
        return IngestionSummary(records=0, details={"series": 0, "values": 0})

    if series_ids:
        selected = {sid for sid in series_ids}
        indicators = [item for item in INDICATORS if item[0] in selected]
    else:
        indicators = list(INDICATORS)

    series_count = 0
    value_count = 0

    async with engine.begin() as conn:
        series_rows = []
        for sid, name, frequency in indicators:
            info = None
            try:
                info = await client.get_series_info(sid)
            except httpx.HTTPStatusError as exc:
                print(f"Skipping FRED series info {sid}: {exc.response.status_code}")
            freq_value = getattr(frequency, "value", None) or str(frequency) if frequency else None
            series_rows.append(
                {
                    "series_id": sid,
                    "name": (info or {}).get("title") or name,
                    "frequency": (info or {}).get("frequency_short")
                    or (info or {}).get("frequency")
                    or freq_value,
                    "unit": (info or {}).get("units_short") or (info or {}).get("units"),
                    "seasonal_adjustment": (info or {}).get("seasonal_adjustment_short")
                    or (info or {}).get("seasonal_adjustment"),
                    "region": (info or {}).get("country"),
                    "data_source": "fred",
                    "metadata": {
                        "indicator": name,
                        "notes": (info or {}).get("notes"),
                        "popularity": (info or {}).get("popularity"),
                        "observation_start": (info or {}).get("observation_start"),
                        "observation_end": (info or {}).get("observation_end"),
                        "last_updated": (info or {}).get("last_updated"),
                    },
                }
            )
            await asyncio.sleep(0.05)
        if series_rows:
            stmt = pg_insert(economic_series).values(series_rows)
            await conn.execute(
                stmt.on_conflict_do_update(
                    index_elements=[economic_series.c.series_id],
                    set_={
                        "name": stmt.excluded.name,
                        "metadata": stmt.excluded.metadata,
                        "updated_at": func.now(),
                    },
                )
            )
            series_count = len(series_rows)

        for sid, _name, _frequency in indicators:
            try:
                observations = await client.get_observations(sid, start, end)
            except httpx.HTTPStatusError as exc:
                print(f"Skipping economic series {sid}: {exc.response.status_code}")
                await asyncio.sleep(0.05)
                continue
            if not observations:
                await asyncio.sleep(0.05)
                continue
            rows: List[Dict] = []
            for obs in observations:
                value = obs.get("value")
                if value in (None, "."):
                    continue
                timestamp = datetime.fromisoformat(obs["date"] + "T00:00:00+00:00")
                rows.append(
                    {
                        "series_id": sid,
                        "timestamp": timestamp,
                        "value": float(value),
                        "revision": None,
                        "metadata": {
                            "realtime_start": obs.get("realtime_start"),
                            "realtime_end": obs.get("realtime_end"),
                        },
                        "ingested_at": datetime.now(tz=UTC),
                    }
                )
            value_count += len(rows)
            for batch in chunk(rows):
                stmt = pg_insert(economic_values).values(batch)
                await conn.execute(
                    stmt.on_conflict_do_update(
                        index_elements=[economic_values.c.series_id, economic_values.c.timestamp],
                        set_={
                            "value": stmt.excluded.value,
                            "revision": stmt.excluded.revision,
                            "metadata": stmt.excluded.metadata,
                            "ingested_at": stmt.excluded.ingested_at,
                        },
                    )
                )
            await asyncio.sleep(0.1)

    await client.close()
    return IngestionSummary(
        records=series_count + value_count,
        details={
            "series": series_count,
            "values": value_count,
        },
    )
