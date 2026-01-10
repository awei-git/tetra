# Factor Pipeline

The factor pipeline computes daily signal features from events, news, economics, and minimal market data. It stores results in `factors.daily_factors` with one row per `symbol + date + factor`.

When no `as_of` date is provided, the pipeline defaults to the current Eastern date to align with end-of-day reporting.

## What’s computed

Signals are normalized to (-1, 1) when surfaced in the API. Event, news, and market factors are cross-sectional z-scores; macro factors use their own time-series z-scores or change rates.

Event factors:
- Counts: `event.count_1d`, `event.count_3d`, `event.count_7d`, `event.count_14d`, `event.count_30d`, `event.count_60d`
- Momentum: `event.momentum_7d`, `event.momentum_30d`
- Intensity: `event.intensity_ratio_7d`, `event.intensity_ratio_14d`
- Type breadth: `event.type_breadth_30d`, `event.type_breadth_90d`
- Type novelty: `event.type_novelty_30d`, `event.type_novelty_90d`
- Importance: `event.importance_high_7d`, `event.importance_high_30d`
- Importance ratios: `event.importance_ratio_7d`, `event.importance_ratio_30d`

News factors:
- Volume: `news.volume_3d`, `news.volume_7d`, `news.volume_14d`, `news.volume_30d`
- Volume momentum: `news.volume_momentum_7d`, `news.volume_momentum_14d`
- Sentiment levels: `news.sentiment_1d`, `news.sentiment_3d`, `news.sentiment_7d`, `news.sentiment_14d`, `news.sentiment_30d`
- Sentiment momentum: `news.sentiment_momentum_3d`, `news.sentiment_momentum_14d`
- Sentiment dispersion: `news.sentiment_dispersion_7d`, `news.sentiment_dispersion_30d`
- Topics: `news.topic_volume_3d`, `news.topic_volume_7d`, `news.topic_velocity_3d`, `news.topic_velocity_7d`

Market controls (minimal):
- Returns: `mkt.return_1d`, `mkt.return_5d`, `mkt.return_10d`, `mkt.return_20d`
- Momentum: `mkt.momentum_60d`, `mkt.momentum_120d`
- Trend: `mkt.sma_20_dist`, `mkt.sma_50_dist`, `mkt.sma_200_dist`, `mkt.ma_cross_50_200`
- Drawdown/breakout: `mkt.drawdown_120d`, `mkt.breakout_252d`
- RSI: `mkt.rsi_14`
- Volatility: `mkt.vol_20d`, `mkt.vol_60d`, `mkt.vol_z_20d`, `mkt.vol_z_60d`
- Volume: `mkt.volume_z_20d`, `mkt.volume_z_60d`

Macro (global, stored under symbol `__macro__`):
- `macro.<SERIES>.z20`, `macro.<SERIES>.z60`
- `macro.<SERIES>.chg20`, `macro.<SERIES>.chg60`
- Series: VIXCLS, DGS10, DGS2, T10Y2Y, BAMLH0A0HYM2, DCOILWTICO

## Tables

Schema: `factors`

- `daily_factors(symbol, as_of, factor, value, source, window_days, metadata, created_at)`
- `factor_runs(id, as_of, run_time, status, summary, error, created_at)`

## API

- `GET /api/factors/summary`
- `GET /api/factors/symbol?symbol=SPY` (includes factor signals + actions)
- `POST /api/factors/selected` (batch symbols with signals + actions)
- `GET /api/factors/alpha` (composite longs/shorts)
- `POST /api/factors/refresh` (optional manual trigger)

## Daily job

Launchd job: `config/launchd/com.tetra.factors-daily.plist`  
Runs at 19:10 local time and writes to `logs/factors-daily.log`.
