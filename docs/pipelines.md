# Pipelines

## Overview
Run commands from the `tetra/` directory.
The data pipelines are grouped by data type and run sequentially to respect API rate limits.

- Market: Polygon (asset metadata + OHLCV)
- Events: Polygon + Finnhub + Alpha Vantage + SEC (earnings, dividends, splits, IPO, macro, filings)
- Economic: FRED (macro series)
- News: Finnhub + NewsAPI (sentiment + macro tags)
- Polymarket: Gamma API (market metadata + snapshot pricing)

For personal usage, start with a small `--symbols` list to control API usage and storage growth.

## Run once

```
python scripts/run_daily_ingest.py
```

Defaults to yesterday -> today. Options:

```
python scripts/run_daily_ingest.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-05 \
  --symbols SPY,QQQ,AAPL \
  --series-ids DGS10,UNRATE \
  --query "ai stocks" \
  --skip-news \
  --skip-polymarket
```

## Automate daily runs
Use the built-in scheduler:

```
python scripts/schedule_daily.py --hour 2 --minute 0
```

Mac/Linux cron example (daily after close at 16:20 local):

```
20 16 * * * cd /path/to/tetra && /usr/bin/python3 scripts/run_daily_ingest.py >> /path/to/tetra_ingest.log 2>&1
```

## Launchd jobs (macOS)
These are the daily automation units tracked in `config/launchd/`:
- `com.tetra.daily-ingest`: data ingestion (16:20 local)
- `com.tetra.factors-daily`: factor computation (19:10 local)
- `com.tetra.gpt-pre`: GPT recommendations pre-market (08:45 local)
- `com.tetra.gpt-post`: GPT recommendations post-close (16:15 local)
- `com.tetra.gpt-challenge`: GPT challenge pass (18:30 local)
- `com.tetra.gpt-factor-review`: GPT critique of factor picks (19:35 local)
- `com.tetra.gpt-summary`: GPT summary of consolidated verdicts (19:45 local)
- `com.tetra.polymarket-history`: weekly Polymarket snapshot (inactive + closed)
- `com.tetra.inference-daily`: inference summary (20:05 local)
- `scripts/run_inference.py`: inference summary (signal IC + event study + Polymarket proxy)

## API keys
Set keys in `tetra/config/secrets.yml`.
If a key is missing, the corresponding pipeline logs a message and returns zero records.

- Polygon: market OHLCV + asset metadata
- Finnhub: earnings calendar + news
- Alpha Vantage: earnings (optional fallback)
- FRED: economic indicators
- NewsAPI: additional news sources
- Polymarket: prediction markets (Gamma API)
