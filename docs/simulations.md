# Simulated Price Paths

This module generates scenario price paths anchored on the latest close for a symbol. Paths are built from historical returns and applied to the most recent price, producing a synthetic time series for comparison.

## Methods

- Historical windows (`historical`)
  - `mode=block`: sample contiguous return windows of length `horizon`.
  - `mode=bootstrap`: resample daily returns with replacement.
- Stress windows (`stress`)
  - Uses historical returns from predefined stress periods.
  - If the stress window is shorter than the horizon, returns are bootstrapped.
- Monte Carlo (`monte_carlo`)
  - Uses historical log-return mean and volatility to simulate a geometric Brownian motion path.

## API

`GET /api/market/simulations`

Query params:
- `symbol` (required): ticker, e.g. `SPY`
- `method`: `historical` | `stress` | `monte_carlo` (default: `historical`)
- `horizon`: number of steps (default: 252)
- `paths`: number of simulated paths (default: 30, max 100)
- `lookback`: number of historical rows for return estimation (default: 2520)
- `stress`: stress window key (default: `covid_2020`)
- `mode`: `block` | `bootstrap` (historical only, default: `block`)
- `seed`: optional integer seed

Response fields:
- `as_of`: timestamp of the latest close used as the starting price
- `start_price`: last close
- `paths_data`: list of simulated paths (each path is a `prices` array)
- `summary`: end-price and end-return percentiles
- `available_stress`: available stress windows for UI selection

## Stress windows

Currently supported (2016+):
- `brexit_2016`: Brexit Referendum Shock (2016-06-20 → 2016-07-08)
- `us_election_2016`: US Election Volatility (2016-11-07 → 2016-11-14)
- `volmageddon_2018`: Volmageddon Spike (2018-02-05 → 2018-02-09)
- `q4_2018`: Q4 2018 Selloff (2018-10-01 → 2018-12-24)
- `trade_war_2019`: US-China Trade War Shock (2019-08-01 → 2019-08-15)
- `repo_2019`: Repo Market Stress (2019-09-16 → 2019-09-30)
- `covid_2020`: COVID Crash (2020-02-19 → 2020-04-30)
- `meme_2021`: Meme Stock Mania (2021-01-25 → 2021-02-02)
- `archegos_2021`: Archegos Unwind (2021-03-22 → 2021-03-31)
- `aug_2021`: Aug 2021 Delta Shock (2021-08-13 → 2021-09-03)
- `evergrande_2021`: Evergrande Stress (2021-09-17 → 2021-10-05)
- `rates_2022`: Rate Shock (2022-01-03 → 2022-10-14)
- `svb_2023`: SVB Banking Shock (2023-03-08 → 2023-03-24)
- `regional_banks_2023`: Regional Bank Stress (2023-04-28 → 2023-05-05)
- `us_downgrade_2023`: US Credit Downgrade Shock (2023-08-01 → 2023-08-18)
- `rates_spike_2023`: Rates Spike (2023-09-21 → 2023-10-27)
- `yen_carry_2024`: Yen Carry Shock (2024-08-05 → 2024-08-23)
- `liberation_day_2025`: Liberation Day Tariff Shock (2025-04-02 → 2025-04-18)

## Example

```bash
curl "http://127.0.0.1:8000/api/market/simulations?symbol=SPY&method=stress&stress=covid_2020&horizon=252&paths=12"
```
