"""Generate simulated price paths from historical returns."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class StressWindow:
    key: str
    label: str
    start: date
    end: date


STRESS_WINDOWS: Dict[str, StressWindow] = {
    "brexit_2016": StressWindow(
        key="brexit_2016",
        label="Brexit Referendum Shock (2016-06-20 → 2016-07-08)",
        start=date(2016, 6, 20),
        end=date(2016, 7, 8),
    ),
    "us_election_2016": StressWindow(
        key="us_election_2016",
        label="US Election Volatility (2016-11-07 → 2016-11-14)",
        start=date(2016, 11, 7),
        end=date(2016, 11, 14),
    ),
    "volmageddon_2018": StressWindow(
        key="volmageddon_2018",
        label="Volmageddon Spike (2018-02-05 → 2018-02-09)",
        start=date(2018, 2, 5),
        end=date(2018, 2, 9),
    ),
    "q4_2018": StressWindow(
        key="q4_2018",
        label="Q4 2018 Selloff (2018-10-01 → 2018-12-24)",
        start=date(2018, 10, 1),
        end=date(2018, 12, 24),
    ),
    "trade_war_2019": StressWindow(
        key="trade_war_2019",
        label="US-China Trade War Shock (2019-08-01 → 2019-08-15)",
        start=date(2019, 8, 1),
        end=date(2019, 8, 15),
    ),
    "repo_2019": StressWindow(
        key="repo_2019",
        label="Repo Market Stress (2019-09-16 → 2019-09-30)",
        start=date(2019, 9, 16),
        end=date(2019, 9, 30),
    ),
    "covid_2020": StressWindow(
        key="covid_2020",
        label="COVID Crash (2020-02-19 → 2020-04-30)",
        start=date(2020, 2, 19),
        end=date(2020, 4, 30),
    ),
    "meme_2021": StressWindow(
        key="meme_2021",
        label="Meme Stock Mania (2021-01-25 → 2021-02-02)",
        start=date(2021, 1, 25),
        end=date(2021, 2, 2),
    ),
    "archegos_2021": StressWindow(
        key="archegos_2021",
        label="Archegos Unwind (2021-03-22 → 2021-03-31)",
        start=date(2021, 3, 22),
        end=date(2021, 3, 31),
    ),
    "aug_2021": StressWindow(
        key="aug_2021",
        label="Aug 2021 Delta Shock (2021-08-13 → 2021-09-03)",
        start=date(2021, 8, 13),
        end=date(2021, 9, 3),
    ),
    "evergrande_2021": StressWindow(
        key="evergrande_2021",
        label="Evergrande Stress (2021-09-17 → 2021-10-05)",
        start=date(2021, 9, 17),
        end=date(2021, 10, 5),
    ),
    "rates_2022": StressWindow(
        key="rates_2022",
        label="Rate Shock (2022-01-03 → 2022-10-14)",
        start=date(2022, 1, 3),
        end=date(2022, 10, 14),
    ),
    "svb_2023": StressWindow(
        key="svb_2023",
        label="SVB Banking Shock (2023-03-08 → 2023-03-24)",
        start=date(2023, 3, 8),
        end=date(2023, 3, 24),
    ),
    "regional_banks_2023": StressWindow(
        key="regional_banks_2023",
        label="Regional Bank Stress (2023-04-28 → 2023-05-05)",
        start=date(2023, 4, 28),
        end=date(2023, 5, 5),
    ),
    "us_downgrade_2023": StressWindow(
        key="us_downgrade_2023",
        label="US Credit Downgrade Shock (2023-08-01 → 2023-08-18)",
        start=date(2023, 8, 1),
        end=date(2023, 8, 18),
    ),
    "rates_spike_2023": StressWindow(
        key="rates_spike_2023",
        label="Rates Spike (2023-09-21 → 2023-10-27)",
        start=date(2023, 9, 21),
        end=date(2023, 10, 27),
    ),
    "yen_carry_2024": StressWindow(
        key="yen_carry_2024",
        label="Yen Carry Shock (2024-08-05 → 2024-08-23)",
        start=date(2024, 8, 5),
        end=date(2024, 8, 23),
    ),
    "liberation_day_2025": StressWindow(
        key="liberation_day_2025",
        label="Liberation Day Tariff Shock (2025-04-02 → 2025-04-18)",
        start=date(2025, 4, 2),
        end=date(2025, 4, 18),
    ),
}


def list_stress_windows() -> List[Dict[str, str]]:
    return [
        {"key": window.key, "label": window.label, "start": window.start.isoformat(), "end": window.end.isoformat()}
        for window in STRESS_WINDOWS.values()
    ]


def compute_log_returns(prices: Sequence[float]) -> List[float]:
    returns: List[float] = []
    for idx in range(1, len(prices)):
        prev = prices[idx - 1]
        curr = prices[idx]
        if prev is None or curr is None or prev <= 0 or curr <= 0:
            continue
        returns.append(math.log(curr / prev))
    return returns


def apply_log_returns(start_price: float, returns: Iterable[float]) -> List[float]:
    prices = [start_price]
    price = start_price
    for step in returns:
        price *= math.exp(step)
        prices.append(price)
    return prices


def _sample_block(returns: Sequence[float], horizon: int, rng: random.Random) -> List[float]:
    if len(returns) < horizon:
        raise ValueError("Not enough returns for requested horizon")
    start_idx = rng.randint(0, len(returns) - horizon)
    return list(returns[start_idx : start_idx + horizon])


def _sample_bootstrap(returns: Sequence[float], horizon: int, rng: random.Random) -> List[float]:
    return [rng.choice(returns) for _ in range(horizon)]


def generate_historical_paths(
    returns: Sequence[float],
    start_price: float,
    horizon: int,
    paths: int,
    mode: str,
    rng: random.Random,
) -> List[List[float]]:
    if not returns:
        raise ValueError("No returns available for historical simulation")
    simulations: List[List[float]] = []
    for _ in range(paths):
        if mode == "bootstrap":
            path_returns = _sample_bootstrap(returns, horizon, rng)
        else:
            path_returns = _sample_block(returns, horizon, rng)
        simulations.append(apply_log_returns(start_price, path_returns))
    return simulations


def generate_stress_paths(
    returns: Sequence[float],
    start_price: float,
    horizon: int,
    paths: int,
    rng: random.Random,
) -> List[List[float]]:
    if not returns:
        raise ValueError("No returns available for stress simulation")
    simulations: List[List[float]] = []
    for _ in range(paths):
        if len(returns) >= horizon:
            path_returns = _sample_block(returns, horizon, rng)
        else:
            path_returns = _sample_bootstrap(returns, horizon, rng)
        simulations.append(apply_log_returns(start_price, path_returns))
    return simulations


def generate_monte_carlo_paths(
    returns: Sequence[float],
    start_price: float,
    horizon: int,
    paths: int,
    rng: random.Random,
) -> List[List[float]]:
    if not returns:
        raise ValueError("No returns available for Monte Carlo simulation")
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / max(1, len(returns) - 1)
    sigma = math.sqrt(variance)
    simulations: List[List[float]] = []
    for _ in range(paths):
        simulated_returns = [rng.gauss(mean, sigma) for _ in range(horizon)]
        simulations.append(apply_log_returns(start_price, simulated_returns))
    return simulations


def summarize_paths(paths: Sequence[Sequence[float]], start_price: float) -> Dict[str, float]:
    if not paths:
        return {}
    end_prices = [path[-1] for path in paths if path]
    if not end_prices:
        return {}
    sorted_prices = sorted(end_prices)
    return {
        "start_price": start_price,
        "min_end": sorted_prices[0],
        "max_end": sorted_prices[-1],
        "median_end": _percentile(sorted_prices, 0.5),
        "p05_end": _percentile(sorted_prices, 0.05),
        "p95_end": _percentile(sorted_prices, 0.95),
        "mean_end": sum(sorted_prices) / len(sorted_prices),
    }


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    pct = max(0.0, min(1.0, pct))
    idx = pct * (len(sorted_values) - 1)
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return sorted_values[lower]
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
