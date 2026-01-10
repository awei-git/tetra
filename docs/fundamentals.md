# Fundamentals Ingestion (Polygon)

This pipeline pulls company financial statements and share counts from Polygon to support Tier‑C / WorldQuant‑style factors.

## Sources

- Polygon financials: `vX/reference/financials`
- Polygon ticker details: `v3/reference/tickers/{ticker}`

## Tables

Schema: `fundamentals`

- `financials` — raw Polygon statement payloads (annual + quarterly)
  - `symbol`, `timeframe`, `fiscal_year`, `fiscal_period`, `period_end`, `filing_date`, `payload`
- `shares` — share counts + market cap snapshots
  - `symbol`, `as_of`, `share_class_shares_outstanding`, `weighted_shares_outstanding`, `market_cap`, `payload`

## Pipeline

`FundamentalsPipeline` runs with daily ingest and stores the latest Polygon payloads for each symbol.

Skip via `scripts/run_daily_ingest.py --skip-fundamentals`.
