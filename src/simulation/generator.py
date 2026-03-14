"""Multi-asset correlated return generator with regime dynamics.

This is the core simulation engine. It generates realistic multi-asset
return paths that capture:

1. Cross-asset correlations (via Cholesky decomposition of covariance)
2. Regime switching (calm → stressed → crisis transitions)
3. Fat tails (Student-t innovations, not just Gaussian)
4. Volatility clustering (GARCH-like persistence within regimes)
5. Mean reversion vs momentum (regime-dependent drift)

Three simulation modes:
- `regime_switching`: Full HMM-driven simulation with regime transitions
- `parametric_stress`: Shock a specific factor and propagate through covariance
- `conditional`: Generate paths conditional on a specific scenario unfolding

All generators produce (paths x horizon x assets) tensors of daily returns.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.simulation.covariance import (
    CovEstimate,
    cholesky,
    nearest_psd,
    annualize_vol,
)
from src.simulation.regime import RegimeModel

# Maximum daily log-return magnitude.  math.exp(0.15) ≈ 16% (up) or -14%
# (down) — exceeds the largest single-day moves in market history while
# preventing compounding blow-up over multi-day scenario horizons.
_MAX_DAILY_LOG_RETURN = 0.15


def _clamp(x: float, lo: float = -_MAX_DAILY_LOG_RETURN, hi: float = _MAX_DAILY_LOG_RETURN) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class SimulationResult:
    """Output of a multi-asset simulation."""
    returns: List[List[List[float]]]   # paths x horizon x assets
    prices: List[List[List[float]]]    # paths x (horizon+1) x assets (cumulative)
    regime_paths: List[List[int]]      # paths x horizon (regime at each step)
    symbols: List[str]
    n_paths: int
    horizon: int
    method: str
    metadata: Dict


# ---------------------------------------------------------------------------
# Random number generation utilities
# ---------------------------------------------------------------------------

def _standard_normal(rng: random.Random) -> float:
    """Standard normal via Box-Muller."""
    u1 = max(rng.random(), 1e-15)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _standard_t(rng: random.Random, df: float) -> float:
    """Student-t random variable via ratio of normal and chi-squared.

    For df > 2, has heavier tails than Gaussian.
    df=5 is a good default for financial returns (matches empirical kurtosis).
    """
    z = _standard_normal(rng)
    # Chi-squared(df) = sum of df standard normals squared
    # Approximate: use gamma distribution via sum
    chi2 = sum(_standard_normal(rng) ** 2 for _ in range(int(df)))
    chi2 = max(chi2, 1e-10)
    return z / math.sqrt(chi2 / df)


def _correlated_innovations(
    L: List[List[float]],
    n_assets: int,
    rng: random.Random,
    df: Optional[float] = None,
) -> List[float]:
    """Generate correlated random innovations via Cholesky decomposition.

    L is the lower Cholesky factor of the covariance matrix.
    If df is set, uses Student-t innovations for fat tails.
    """
    if df is not None and df > 2:
        z = [_standard_t(rng, df) for _ in range(n_assets)]
        # Scale so variance matches: Var(t_df) = df/(df-2)
        scale = math.sqrt((df - 2) / df)
        z = [zi * scale for zi in z]
    else:
        z = [_standard_normal(rng) for _ in range(n_assets)]

    # Correlate: x = L @ z
    return [sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(n_assets)]


# ---------------------------------------------------------------------------
# 1. Regime-switching simulation
# ---------------------------------------------------------------------------

def simulate_regime_switching(
    regime_model: RegimeModel,
    regime_covs: Dict[int, CovEstimate],
    start_prices: List[float],
    symbols: List[str],
    horizon: int = 60,
    n_paths: int = 1000,
    df: Optional[float] = 5.0,
    vol_persistence: float = 0.85,
    seed: int = 42,
) -> SimulationResult:
    """Simulate multi-asset returns with regime-switching dynamics.

    At each timestep:
    1. Current regime determines covariance structure
    2. Regime transitions follow the fitted transition matrix
    3. Innovations are drawn from correlated Student-t (fat tails)
    4. Volatility has GARCH-like persistence within regime

    Args:
        regime_model: fitted HMM model
        regime_covs: covariance estimate per regime (from regime_conditional_cov)
        start_prices: initial prices per asset
        symbols: asset labels
        horizon: simulation horizon in trading days
        n_paths: number of paths to generate
        df: Student-t degrees of freedom (None = Gaussian)
        vol_persistence: GARCH-like vol persistence (0 = no memory, 1 = full memory)
        seed: random seed
    """
    rng = random.Random(seed)
    n_assets = len(symbols)
    k = regime_model.n_states

    # Precompute Cholesky factors per regime
    chol_factors: Dict[int, List[List[float]]] = {}
    regime_drifts: Dict[int, List[float]] = {}
    for regime_id, cov_est in regime_covs.items():
        chol_factors[regime_id] = cholesky(nearest_psd(cov_est.cov))
        # Drift: use regime mean return (small, daily scale)
        if regime_id < len(regime_model.means):
            # Mean return is feature index 1
            daily_drift = regime_model.means[regime_id][1] if len(regime_model.means[regime_id]) > 1 else 0.0
            regime_drifts[regime_id] = [daily_drift] * n_assets
        else:
            regime_drifts[regime_id] = [0.0] * n_assets

    # Transition matrix
    trans = regime_model.transition_matrix

    # Start from current regime distribution
    current_probs = regime_model.regime_probs[-1] if regime_model.regime_probs else regime_model.initial_probs

    all_returns = []
    all_prices = []
    all_regimes = []

    for _ in range(n_paths):
        # Sample initial regime
        r = rng.random()
        cumulative = 0.0
        current_regime = 0
        for s in range(k):
            cumulative += current_probs[s]
            if r <= cumulative:
                current_regime = s
                break

        path_returns = []
        path_prices = [start_prices[:]]
        path_regimes = []
        vol_scale = 1.0  # GARCH-like vol scaling

        for t in range(horizon):
            # Regime transition
            r = rng.random()
            cumulative = 0.0
            for s in range(k):
                cumulative += trans[current_regime][s]
                if r <= cumulative:
                    current_regime = s
                    break

            path_regimes.append(current_regime)

            # Get covariance for current regime
            if current_regime not in chol_factors:
                # Fallback to first available
                current_regime = next(iter(chol_factors))

            L = chol_factors[current_regime]
            drift = regime_drifts.get(current_regime, [0.0] * n_assets)

            # Generate correlated innovations
            innovations = _correlated_innovations(L, n_assets, rng, df)

            # Apply GARCH-like vol persistence
            daily_returns = [
                drift[i] + innovations[i] * vol_scale
                for i in range(n_assets)
            ]

            # Update vol scale (mean-reverting to 1.0, capped to prevent runaway)
            realized_vol = math.sqrt(sum(r ** 2 for r in daily_returns) / n_assets)
            expected_vol = math.sqrt(sum(L[i][i] ** 2 for i in range(n_assets)) / n_assets)
            if expected_vol > 0:
                vol_ratio = realized_vol / expected_vol
                vol_scale = vol_persistence * vol_scale + (1 - vol_persistence) * vol_ratio
                vol_scale = max(0.3, min(vol_scale, 3.0))

            daily_returns = [_clamp(r) for r in daily_returns]
            path_returns.append(daily_returns)

            # Update prices
            new_prices = [
                path_prices[-1][i] * math.exp(daily_returns[i])
                for i in range(n_assets)
            ]
            path_prices.append(new_prices)

        all_returns.append(path_returns)
        all_prices.append(path_prices)
        all_regimes.append(path_regimes)

    return SimulationResult(
        returns=all_returns,
        prices=all_prices,
        regime_paths=all_regimes,
        symbols=symbols,
        n_paths=n_paths,
        horizon=horizon,
        method="regime_switching",
        metadata={
            "df": df,
            "vol_persistence": vol_persistence,
            "n_regimes": k,
        },
    )


# ---------------------------------------------------------------------------
# 2. Parametric stress simulation
# ---------------------------------------------------------------------------

def simulate_factor_shock(
    base_cov: CovEstimate,
    start_prices: List[float],
    symbols: List[str],
    shock_symbol: str,
    shock_magnitude: float,
    horizon: int = 20,
    n_paths: int = 500,
    df: Optional[float] = 5.0,
    decay_rate: float = 0.1,
    seed: int = 42,
) -> SimulationResult:
    """Simulate a shock to one asset and its propagation through correlations.

    The shock propagates via the covariance structure:
    - Directly correlated assets move immediately
    - The shock decays exponentially over the horizon
    - Volatility increases post-shock (vol-of-vol effect)

    Args:
        base_cov: baseline covariance estimate
        start_prices: initial prices
        symbols: asset labels
        shock_symbol: which asset gets shocked
        shock_magnitude: shock size in daily return units (e.g., -0.05 = -5% day)
        horizon: days to simulate after shock
        n_paths: number of paths
        df: Student-t degrees of freedom
        decay_rate: how fast the shock decays (higher = faster)
    """
    rng = random.Random(seed)
    n = len(symbols)

    if shock_symbol not in symbols:
        raise ValueError(f"shock_symbol '{shock_symbol}' not in symbols")

    shock_idx = symbols.index(shock_symbol)
    L = cholesky(nearest_psd(base_cov.cov))

    # Compute conditional shock propagation via covariance
    # E[X_j | X_i = shock] = cov(j,i)/var(i) * shock
    var_i = base_cov.cov[shock_idx][shock_idx]
    conditional_means = [
        base_cov.cov[j][shock_idx] / max(var_i, 1e-12) * shock_magnitude
        for j in range(n)
    ]

    all_returns = []
    all_prices = []

    for _ in range(n_paths):
        path_returns = []
        path_prices = [start_prices[:]]

        for t in range(horizon):
            # Shock decays exponentially
            shock_factor = math.exp(-decay_rate * t)

            # Base innovation
            innovations = _correlated_innovations(L, n, rng, df)

            # Add decaying conditional mean from shock
            daily_returns = [
                innovations[i] + conditional_means[i] * shock_factor
                for i in range(n)
            ]

            # Post-shock vol increase (first few days)
            if t < 5:
                vol_bump = 1.0 + abs(shock_magnitude) * (5 - t) / 5
                daily_returns = [r * vol_bump for r in daily_returns]

            daily_returns = [_clamp(r) for r in daily_returns]
            path_returns.append(daily_returns)
            new_prices = [
                path_prices[-1][i] * math.exp(daily_returns[i])
                for i in range(n)
            ]
            path_prices.append(new_prices)

        all_returns.append(path_returns)
        all_prices.append(path_prices)

    return SimulationResult(
        returns=all_returns,
        prices=all_prices,
        regime_paths=[[0] * horizon] * n_paths,
        symbols=symbols,
        n_paths=n_paths,
        horizon=horizon,
        method="factor_shock",
        metadata={
            "shock_symbol": shock_symbol,
            "shock_magnitude": shock_magnitude,
            "decay_rate": decay_rate,
        },
    )


# ---------------------------------------------------------------------------
# 3. Conditional scenario simulation
# ---------------------------------------------------------------------------

def simulate_scenario(
    base_cov: CovEstimate,
    start_prices: List[float],
    symbols: List[str],
    target_moves: Dict[str, float],
    move_horizon: int = 20,
    n_paths: int = 500,
    df: Optional[float] = 5.0,
    seed: int = 42,
) -> SimulationResult:
    """Simulate paths conditional on specific asset moves occurring.

    Unlike factor_shock (which is instant), this generates paths where
    certain assets reach target returns by move_horizon, while other
    assets evolve consistently with the correlation structure.

    Uses drift adjustment: assets with targets get drift = target_return / horizon,
    while non-target assets get conditional drift from covariance.

    Args:
        base_cov: baseline covariance
        start_prices: initial prices
        symbols: asset labels
        target_moves: dict of symbol → target cumulative return (e.g., {"AAPL": -0.10})
        move_horizon: days over which targets are achieved
        n_paths: number of paths
        df: Student-t degrees of freedom
    """
    rng = random.Random(seed)
    n = len(symbols)
    L = cholesky(nearest_psd(base_cov.cov))

    # Compute daily drift for target assets
    # Cap daily drift at ±5% to keep scenarios realistic
    _MAX_DAILY_DRIFT = 0.05
    daily_drifts = [0.0] * n
    target_indices = set()
    for sym, target_ret in target_moves.items():
        if sym in symbols:
            idx = symbols.index(sym)
            drift = target_ret / move_horizon
            daily_drifts[idx] = max(-_MAX_DAILY_DRIFT, min(_MAX_DAILY_DRIFT, drift))
            target_indices.add(idx)

    # Conditional drift for non-target assets
    # E[X_j | X_i targets] = sum_i cov(j,i)/var(i) * drift_i
    # Clamp beta (cov/var) to prevent blow-up when variance is tiny
    for j in range(n):
        if j in target_indices:
            continue
        cond_drift = 0.0
        for i in target_indices:
            var_i = base_cov.cov[i][i]
            if var_i > 1e-10:
                beta = base_cov.cov[j][i] / var_i
                beta = max(-5.0, min(5.0, beta))
                cond_drift += beta * daily_drifts[i]
        daily_drifts[j] = _clamp(cond_drift)

    all_returns = []
    all_prices = []

    # Cap cumulative log-return per asset to ±100% (e^1 ≈ 2.7x or 0.37x)
    _MAX_CUM_LOG_RETURN = 1.0

    for _ in range(n_paths):
        path_returns = []
        path_prices = [start_prices[:]]
        cum_log_ret = [0.0] * n

        for t in range(move_horizon):
            innovations = _correlated_innovations(L, n, rng, df)
            daily_returns = []
            for i in range(n):
                r = _clamp(daily_drifts[i] + innovations[i])
                # Freeze asset if cumulative return already at limit
                headroom_up = _MAX_CUM_LOG_RETURN - cum_log_ret[i]
                headroom_dn = -_MAX_CUM_LOG_RETURN - cum_log_ret[i]
                r = max(headroom_dn, min(headroom_up, r))
                cum_log_ret[i] += r
                daily_returns.append(r)
            path_returns.append(daily_returns)

            new_prices = [
                path_prices[-1][i] * math.exp(daily_returns[i])
                for i in range(n)
            ]
            path_prices.append(new_prices)

        all_returns.append(path_returns)
        all_prices.append(path_prices)

    return SimulationResult(
        returns=all_returns,
        prices=all_prices,
        regime_paths=[[0] * move_horizon] * n_paths,
        symbols=symbols,
        n_paths=n_paths,
        horizon=move_horizon,
        method="conditional_scenario",
        metadata={
            "target_moves": target_moves,
        },
    )


# ---------------------------------------------------------------------------
# Summary statistics for simulation results
# ---------------------------------------------------------------------------

def summarize_simulation(result: SimulationResult) -> Dict:
    """Compute summary statistics from simulation output."""
    n = len(result.symbols)
    summary = {"method": result.method, "n_paths": result.n_paths, "horizon": result.horizon}

    # Per-asset statistics at horizon end
    asset_stats = []
    for j in range(n):
        end_prices = [
            result.prices[p][-1][j]
            for p in range(result.n_paths)
        ]
        start = result.prices[0][0][j]
        returns = [(ep / start - 1.0) for ep in end_prices]
        returns.sort()

        # Max drawdown per path
        drawdowns = []
        for p in range(result.n_paths):
            peak = result.prices[p][0][j]
            max_dd = 0.0
            for t in range(1, len(result.prices[p])):
                peak = max(peak, result.prices[p][t][j])
                dd = (result.prices[p][t][j] - peak) / peak if peak > 0 else 0
                max_dd = min(max_dd, dd)
            drawdowns.append(max_dd)

        n_r = len(returns)
        asset_stats.append({
            "symbol": result.symbols[j],
            "mean_return": sum(returns) / n_r,
            "median_return": returns[n_r // 2],
            "p05": returns[max(0, int(n_r * 0.05))],
            "p25": returns[max(0, int(n_r * 0.25))],
            "p75": returns[min(n_r - 1, int(n_r * 0.75))],
            "p95": returns[min(n_r - 1, int(n_r * 0.95))],
            "worst": returns[0],
            "best": returns[-1],
            "prob_negative": sum(1 for r in returns if r < 0) / n_r,
            "mean_max_drawdown": sum(drawdowns) / len(drawdowns),
            "worst_drawdown": min(drawdowns),
        })

    summary["assets"] = asset_stats

    # Regime statistics (if available)
    if result.regime_paths and any(any(r != 0 for r in path) for path in result.regime_paths):
        regime_counts: Dict[int, int] = {}
        total = 0
        for path in result.regime_paths:
            for r in path:
                regime_counts[r] = regime_counts.get(r, 0) + 1
                total += 1
        summary["regime_distribution"] = {
            r: count / total for r, count in sorted(regime_counts.items())
        }

    return summary
