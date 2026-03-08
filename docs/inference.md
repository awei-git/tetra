# Inference Pipeline

This pipeline summarizes the predictive power of signals and market reactions.

## Outputs
### 1) Signal Quality Leaderboard
Stores cross-sectional Spearman IC for each factor vs forward returns.

Table: `inference.signal_leaderboard`
- `factor`, `horizon_days`, `avg_ic`, `median_ic`, `hit_rate`, `days`, `observations`

### 2) Event Impact Study
Average returns around events (earnings/filings/dividends/etc).

Table: `inference.event_study`
- `event_type`, `window_days`, `avg_return`, `median_return`, `observations`

### 3) Polymarket Calibration (Proxy)
Uses closed markets with a dominant outcome price (≥ 0.9) as proxy resolution.

Tables:
- `inference.polymarket_summary`
- `inference.polymarket_bins`

## Run manually
```
python scripts/run_inference.py
```

## Notes
- Signal IC uses the last 180 trading days by default.
- Event study uses the last 365 days of events.
- Polymarket calibration is proxy-only until resolved outcomes are available from CLOB/UMA feeds.
