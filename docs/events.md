# Events

## Covered event types
The event pipeline stores everything in `event.events` with `event_type`, `event_time`, `source`, and full raw `payload`.

- Earnings: Polygon + Finnhub + Alpha Vantage
- Dividends: Polygon
- Splits: Polygon
- IPO calendar: Finnhub
- Macro calendar: Finnhub economic calendar (country, impact, actual/forecast)
- SEC filings: EDGAR submissions (10-K, 10-Q, 8-K, S-1, etc.)

## Notes
- `symbol` is populated for company events; macro events use `NULL`.
- `importance` is provider-specific (earnings time, cash amount, split ratio, impact level).
- Deduping is based on (`source`, `external_id`, `event_time`).
- SEC requires a `sec_user_agent` set in `config/secrets.yml` (email contact is recommended).
- Finnhub economic calendar may return 403 on free tiers; the pipeline skips it if blocked.

## Run manually

```
python scripts/run_daily_ingest.py --start-date 2024-01-01 --end-date 2024-01-31
```

The event pipeline respects the date range for all sources.
