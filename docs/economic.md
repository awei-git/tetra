# Economic Data

## Sources
- FRED (Federal Reserve Economic Data)

## Series catalog
The series list is defined in `src/definitions/economic_indicators.py` and includes
rates, inflation, labor, growth, housing, and market indicators.

## Stored tables
- `economic.series`: catalog + metadata (title, frequency, units, notes)
- `economic.values`: observations

## Backfill example (10 years)

```
python scripts/run_daily_ingest.py \
  --start-date 2016-01-07 \
  --end-date 2026-01-07 \
  --skip-market \
  --skip-events \
  --skip-news
```

## Notes
- Set `fred` API key in `config/secrets.yml`.
- If FRED metadata calls fail, the pipeline still loads observations.
