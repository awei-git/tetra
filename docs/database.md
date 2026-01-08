# Database

## Recommended local setup
For personal usage, a single Postgres instance in Docker is the simplest and most reliable option. It keeps the data durable with a local volume and avoids extra infra.

Start the database from `tetra/`:

```
docker compose up -d
```

Initialize schemas + tables:

```
python scripts/init_db.py
```

## Schemas
- `market`: assets + OHLCV bars
- `event`: earnings and related events
- `economic`: FRED series + observations
- `news`: articles + metadata

## Key tables
- `market.assets`: asset metadata
- `market.ohlcv`: daily OHLCV
- `event.events`: earnings/events
- `economic.series`: series catalog
- `economic.values`: observations
- `news.articles`: news articles

## Upserts and uniqueness
Event and news upserts rely on unique indexes:
- `event.events` on (`source`, `external_id`, `event_time`)
- `news.articles` on (`source`, `external_id`, `published_at`)

`python scripts/init_db.py` creates these indexes if missing.

## Connection settings
Update `tetra/config/secrets.yml` for DB password. Defaults in `tetra/docker-compose.yml`:
- DB: `tetra`
- User: `tetra_user`
- Password: `tetra_password`
