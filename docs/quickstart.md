# Quickstart

This guide sets up the database, pipelines, and UI for local personal use.
Run the commands from the `tetra/` directory unless noted.

## Prereqs
- Python 3.12+
- Docker Desktop

## 1) Configure secrets
Copy the template and add API keys:

```
cp config/secrets.example.yml config/secrets.yml
```

Fill in:
- Polygon (market OHLCV + asset metadata)
- FRED (economic series)
- Finnhub (events + news)
- NewsAPI (news)
- Alpha Vantage (events, optional)
- SEC user agent (filings; include a contact email)

## 2) Start Postgres
From `tetra/`:

```
docker compose up -d
```

## 3) Initialize schemas + tables

```
python scripts/init_db.py
```

## 4) Run pipelines once

```
python scripts/run_daily_ingest.py --start-date 2024-01-01 --end-date 2024-01-05
```

To limit usage while testing:

```
python scripts/run_daily_ingest.py --symbols SPY,QQQ --series-ids DGS10,UNRATE --query "stocks"
```

## 5) Start the API + UI

```
uvicorn src.api.app:app --reload
```

Open `http://localhost:8000`.

## 6) Schedule daily ingestion

```
python scripts/schedule_daily.py --hour 2 --minute 0
```

Use `--once` for a single run.
