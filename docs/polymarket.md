# Polymarket Data

## What we ingest
The Polymarket pipeline pulls market metadata and snapshot pricing from the public Gamma API.

Stored fields include:
- Market metadata: question, slug, category, description, end time, created time
- Market status: active/closed/archived
- Pricing snapshot: best bid/ask, volume, liquidity
- Identifiers: condition ID + CLOB token IDs
- Full raw payloads for traceability

## Tables
`polymarket.markets`
- One row per market ID (upserted on each run).

`polymarket.snapshots`
- One row per market ID + snapshot time.
- Captures pricing + status at run time for time series analysis.

## Pipeline
`PolymarketPipeline` runs inside the daily ingest sequence and writes the tables above.
For historical coverage (inactive/closed markets), run a weekly job.

Skip it if needed:
```
python scripts/run_daily_ingest.py --skip-polymarket
```

Run historical backfill (inactive + closed):
```
python scripts/run_polymarket_ingest.py --include-inactive
```

## API Notes
Gamma API is public but requires a User-Agent header to avoid 403s.
The CLOB API (order books/trades) is not yet ingested; it can be added using the Polymarket API keys.
