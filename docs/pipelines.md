# Pipelines

## Overview
Run commands from the `tetra/` directory.
The data pipelines are grouped by data type and run sequentially to respect API rate limits.

- Market: Polygon (asset metadata + OHLCV)
- Events: Polygon + Finnhub + Alpha Vantage + SEC (earnings, dividends, splits, IPO, macro, filings)
- Economic: FRED (macro series)
- News: Finnhub + NewsAPI (sentiment + macro tags)

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
  --skip-news
```

## Automate daily runs
Use the built-in scheduler:

```
python scripts/schedule_daily.py --hour 2 --minute 0
```

Mac/Linux cron example (daily at 02:00):

```
0 2 * * * cd /path/to/tetra && /usr/bin/python3 scripts/run_daily_ingest.py >> /path/to/tetra_ingest.log 2>&1
```

## API keys
Set keys in `tetra/config/secrets.yml`.
If a key is missing, the corresponding pipeline logs a message and returns zero records.

- Polygon: market OHLCV + asset metadata
- Finnhub: earnings calendar + news
- Alpha Vantage: earnings (optional fallback)
- FRED: economic indicators
- NewsAPI: additional news sources
