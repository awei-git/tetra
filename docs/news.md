# News Data

## Sources
- Finnhub (general + company news)
- NewsAPI (general articles)

## Sentiment
- Sentiment is computed per article using VADER and stored in:
  - `news.articles.sentiment` (compound score)
  - `news.articles.sentiment_confidence` (absolute score)
- The raw payload also includes `analysis` with sentiment label + macro topics.

## Macro topics
A simple keyword tagger marks macro themes (inflation, rates, growth, labor, housing, energy, FX, geopolitics).

## API limits
Most news APIs do not provide 10 years of history on standard tiers.
Expect shorter coverage (days to months depending on provider).

## Backfill example

```
python scripts/run_daily_ingest.py \
  --start-date 2025-01-01 \
  --end-date 2026-01-07 \
  --skip-market \
  --skip-events \
  --skip-economic
```
