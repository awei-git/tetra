"""Walk-forward backtesting engine for signal validation.

This is NOT a traditional backtest that runs a strategy on historical data.
It's a signal validation framework that answers:

1. Does this signal predict forward returns? (IC analysis)
2. How fast does the alpha decay? (decay curve)
3. Is the signal stable or was it a fluke? (rolling IC, bootstrap confidence)
4. Does the signal work in all regimes or only some? (regime-conditional IC)
5. How much of the signal is redundant with other signals? (orthogonalization)

Walk-forward protocol:
- At each rebalance date, use ONLY data available at that point
- No lookahead: all statistics computed on trailing windows
- Expanding or rolling training window (configurable)
- Out-of-sample evaluation on the next period
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SignalEvaluation:
    """Evaluation of a single signal's predictive power."""
    signal_name: str
    horizon_days: int
    # IC statistics
    mean_ic: float               # average rank IC (Spearman)
    median_ic: float
    ic_std: float                # IC volatility (lower = more stable)
    ic_ir: float                 # IC information ratio: mean_ic / ic_std
    hit_rate: float              # fraction of periods with positive IC
    # Decay analysis
    ic_by_day: List[float]       # IC at each forward day (decay curve)
    half_life_days: Optional[float]  # days until IC halves
    # Stability
    ic_by_period: List[Tuple[str, float]]  # (period_label, IC) for rolling windows
    bootstrap_ci_95: Tuple[float, float]   # 95% confidence interval on mean IC
    # Regime
    ic_by_regime: Dict[str, float]  # regime_label → IC
    # Practical
    turnover: float              # average daily turnover if used as signal
    correlation_with: Dict[str, float]  # correlation with other signals


@dataclass
class BacktestResult:
    """Full backtest output."""
    signals: List[SignalEvaluation]
    combined_ic: float           # IC of optimally combined signal
    regime_labels: List[str]     # regime at each evaluation date
    evaluation_dates: List[str]  # dates used for evaluation
    n_assets: int
    n_periods: int
    methodology: str


# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------

def _rank(values: List[float]) -> List[float]:
    """Fractional rank (handles ties)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(x: List[float], y: List[float]) -> Optional[float]:
    """Spearman rank correlation."""
    if len(x) != len(y) or len(x) < 3:
        return None
    return _pearson(_rank(x), _rank(y))


def _pearson(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 3:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def _bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - confidence) / 2
    lo = means[max(0, int(alpha * n_boot))]
    hi = means[min(n_boot - 1, int((1 - alpha) * n_boot))]
    return (lo, hi)


# ---------------------------------------------------------------------------
# Walk-forward signal evaluation
# ---------------------------------------------------------------------------

def evaluate_signal(
    signal_values: Dict[str, List[Tuple[str, float]]],
    forward_returns: Dict[str, List[Tuple[str, float]]],
    signal_name: str,
    horizons: Sequence[int] = (1, 5, 10, 20),
    min_assets: int = 10,
    regime_labels: Optional[Dict[str, str]] = None,
) -> SignalEvaluation:
    """Evaluate a single signal's predictive power via walk-forward IC analysis.

    Args:
        signal_values: symbol → [(date_str, signal_value), ...] sorted by date
        forward_returns: symbol → [(date_str, forward_return), ...] for each horizon day
        signal_name: label for this signal
        horizons: forward return horizons to evaluate (days)
        min_assets: minimum cross-sectional observations per date
        regime_labels: date_str → regime_label (optional)

    The protocol:
    - For each date, collect (signal, forward_return) pairs across all assets
    - Compute rank IC (Spearman correlation)
    - No lookahead: signal at date t predicts return from t to t+horizon
    """
    # Build date-aligned cross-sections
    # For each date: collect {symbol: signal_value}
    signal_by_date: Dict[str, Dict[str, float]] = {}
    for symbol, pairs in signal_values.items():
        for date_str, val in pairs:
            signal_by_date.setdefault(date_str, {})[symbol] = val

    return_by_date: Dict[str, Dict[str, float]] = {}
    for symbol, pairs in forward_returns.items():
        for date_str, val in pairs:
            return_by_date.setdefault(date_str, {})[symbol] = val

    # Dates where we have both signal and returns
    common_dates = sorted(set(signal_by_date.keys()) & set(return_by_date.keys()))

    # Compute IC for each date
    daily_ics: List[Tuple[str, float]] = []
    for date_str in common_dates:
        sig_map = signal_by_date[date_str]
        ret_map = return_by_date[date_str]
        common_syms = sorted(set(sig_map.keys()) & set(ret_map.keys()))
        if len(common_syms) < min_assets:
            continue
        signals = [sig_map[s] for s in common_syms]
        returns = [ret_map[s] for s in common_syms]
        ic = _spearman(signals, returns)
        if ic is not None:
            daily_ics.append((date_str, ic))

    if not daily_ics:
        return _empty_eval(signal_name, horizons[0] if horizons else 1)

    ic_values = [ic for _, ic in daily_ics]
    mean_ic = sum(ic_values) / len(ic_values)
    ic_std = math.sqrt(sum((v - mean_ic) ** 2 for v in ic_values) / max(len(ic_values) - 1, 1))
    median_ic = sorted(ic_values)[len(ic_values) // 2]
    ic_ir = mean_ic / ic_std if ic_std > 0 else 0.0
    hit_rate = sum(1 for v in ic_values if v > 0) / len(ic_values)

    # Bootstrap CI
    ci = _bootstrap_ci(ic_values)

    # IC by regime
    ic_by_regime: Dict[str, float] = {}
    if regime_labels:
        regime_ics: Dict[str, List[float]] = {}
        for date_str, ic in daily_ics:
            regime = regime_labels.get(date_str, "unknown")
            regime_ics.setdefault(regime, []).append(ic)
        for regime, ics in regime_ics.items():
            ic_by_regime[regime] = sum(ics) / len(ics)

    # IC decay curve (placeholder — needs multi-horizon forward returns)
    ic_by_day = [mean_ic]  # single horizon for now

    # Half-life estimation
    half_life = None
    if len(ic_by_day) > 1 and ic_by_day[0] > 0:
        target = ic_by_day[0] / 2
        for d, ic in enumerate(ic_by_day):
            if ic <= target:
                half_life = float(d)
                break

    # Turnover: average absolute change in signal ranks between dates
    turnover = _estimate_turnover(signal_by_date, common_dates)

    # Rolling IC for stability
    window = max(20, len(daily_ics) // 10)
    ic_by_period = []
    for start in range(0, len(daily_ics) - window + 1, window):
        chunk = daily_ics[start:start + window]
        period_label = f"{chunk[0][0]}_{chunk[-1][0]}"
        period_ic = sum(ic for _, ic in chunk) / len(chunk)
        ic_by_period.append((period_label, period_ic))

    return SignalEvaluation(
        signal_name=signal_name,
        horizon_days=horizons[0] if horizons else 1,
        mean_ic=mean_ic,
        median_ic=median_ic,
        ic_std=ic_std,
        ic_ir=ic_ir,
        hit_rate=hit_rate,
        ic_by_day=ic_by_day,
        half_life_days=half_life,
        ic_by_period=ic_by_period,
        bootstrap_ci_95=ci,
        ic_by_regime=ic_by_regime,
        turnover=turnover,
        correlation_with={},
    )


def _estimate_turnover(
    signal_by_date: Dict[str, Dict[str, float]],
    dates: List[str],
) -> float:
    """Estimate signal turnover as average rank change between consecutive dates."""
    if len(dates) < 2:
        return 0.0

    turnovers = []
    for i in range(1, len(dates)):
        prev = signal_by_date.get(dates[i - 1], {})
        curr = signal_by_date.get(dates[i], {})
        common = sorted(set(prev.keys()) & set(curr.keys()))
        if len(common) < 5:
            continue
        prev_ranks = _rank([prev[s] for s in common])
        curr_ranks = _rank([curr[s] for s in common])
        # Turnover = mean absolute rank change / N
        n = len(common)
        rank_change = sum(abs(curr_ranks[j] - prev_ranks[j]) for j in range(n)) / n
        turnovers.append(rank_change / n)  # normalize by universe size

    return sum(turnovers) / len(turnovers) if turnovers else 0.0


def _empty_eval(signal_name: str, horizon: int) -> SignalEvaluation:
    return SignalEvaluation(
        signal_name=signal_name, horizon_days=horizon,
        mean_ic=0.0, median_ic=0.0, ic_std=0.0, ic_ir=0.0, hit_rate=0.0,
        ic_by_day=[], half_life_days=None, ic_by_period=[],
        bootstrap_ci_95=(0.0, 0.0), ic_by_regime={}, turnover=0.0,
        correlation_with={},
    )


# ---------------------------------------------------------------------------
# Multi-signal evaluation and combination
# ---------------------------------------------------------------------------

def evaluate_signal_set(
    signals: Dict[str, Dict[str, List[Tuple[str, float]]]],
    forward_returns: Dict[str, List[Tuple[str, float]]],
    horizons: Sequence[int] = (1, 5, 20),
    min_assets: int = 10,
    regime_labels: Optional[Dict[str, str]] = None,
) -> BacktestResult:
    """Evaluate multiple signals and their optimal combination.

    Args:
        signals: signal_name → {symbol → [(date, value), ...]}
        forward_returns: symbol → [(date, return), ...]
        horizons: evaluation horizons
        min_assets: minimum cross-section size
        regime_labels: date → regime label

    Returns BacktestResult with per-signal evaluations and combined IC.
    """
    evals = []
    for sig_name, sig_data in signals.items():
        ev = evaluate_signal(
            signal_values=sig_data,
            forward_returns=forward_returns,
            signal_name=sig_name,
            horizons=horizons,
            min_assets=min_assets,
            regime_labels=regime_labels,
        )
        evals.append(ev)

    # Compute signal correlations
    _compute_signal_correlations(evals, signals)

    # Combined IC: IC-weighted combination
    total_weight = 0.0
    combined_ic = 0.0
    for ev in evals:
        if ev.ic_ir > 0:
            weight = ev.ic_ir  # weight by information ratio
            combined_ic += ev.mean_ic * weight
            total_weight += weight
    if total_weight > 0:
        combined_ic /= total_weight

    # Collect evaluation dates and regime labels
    all_dates = set()
    for sig_data in signals.values():
        for sym_data in sig_data.values():
            for date_str, _ in sym_data:
                all_dates.add(date_str)

    eval_dates = sorted(all_dates)
    regimes = [regime_labels.get(d, "unknown") for d in eval_dates] if regime_labels else []

    return BacktestResult(
        signals=evals,
        combined_ic=combined_ic,
        regime_labels=regimes,
        evaluation_dates=eval_dates,
        n_assets=len(forward_returns),
        n_periods=len(eval_dates),
        methodology="walk_forward_rank_ic",
    )


def _compute_signal_correlations(
    evals: List[SignalEvaluation],
    signals: Dict[str, Dict[str, List[Tuple[str, float]]]],
) -> None:
    """Compute pairwise correlations between signal IC series."""
    # Build IC series per signal
    # (simplified: compute correlation of signal values across common dates/symbols)
    sig_names = [ev.signal_name for ev in evals]
    for i, ev_i in enumerate(evals):
        for j, ev_j in enumerate(evals):
            if i == j:
                continue
            # Use IC-by-period correlation as proxy
            if ev_i.ic_by_period and ev_j.ic_by_period:
                common_periods = min(len(ev_i.ic_by_period), len(ev_j.ic_by_period))
                if common_periods >= 3:
                    x = [ev_i.ic_by_period[k][1] for k in range(common_periods)]
                    y = [ev_j.ic_by_period[k][1] for k in range(common_periods)]
                    corr = _pearson(x, y)
                    if corr is not None:
                        ev_i.correlation_with[ev_j.signal_name] = corr


# ---------------------------------------------------------------------------
# Strategy backtest (portfolio-level)
# ---------------------------------------------------------------------------

@dataclass
class PortfolioBacktestResult:
    """Results from a portfolio-level backtest."""
    dates: List[str]
    daily_returns: List[float]
    cumulative_returns: List[float]
    positions: List[Dict[str, float]]  # date → {symbol: weight}
    # Performance metrics
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    avg_turnover: float
    # Risk
    var_95: float
    cvar_95: float
    worst_day: float
    best_day: float
    skewness: float
    kurtosis: float


def backtest_signal_strategy(
    signal_values: Dict[str, List[Tuple[str, float]]],
    asset_returns: Dict[str, List[Tuple[str, float]]],
    top_n: int = 10,
    rebalance_freq: int = 5,
    long_only: bool = True,
    transaction_cost_bps: float = 10.0,
) -> PortfolioBacktestResult:
    """Backtest a simple signal-based strategy.

    At each rebalance date:
    - Rank assets by signal
    - Go long top_n (and short bottom_n if long_only=False)
    - Equal weight within long/short buckets
    - Deduct transaction costs on turnover

    This is a VALIDATION tool, not a production strategy.
    It answers: "If I traded purely on this signal, what would happen?"
    """
    # Build date-aligned data
    all_dates = set()
    for pairs in signal_values.values():
        for d, _ in pairs:
            all_dates.add(d)
    for pairs in asset_returns.values():
        for d, _ in pairs:
            all_dates.add(d)
    dates = sorted(all_dates)

    sig_by_date: Dict[str, Dict[str, float]] = {}
    for sym, pairs in signal_values.items():
        for d, v in pairs:
            sig_by_date.setdefault(d, {})[sym] = v

    ret_by_date: Dict[str, Dict[str, float]] = {}
    for sym, pairs in asset_returns.items():
        for d, v in pairs:
            ret_by_date.setdefault(d, {})[sym] = v

    # Run backtest
    daily_returns = []
    cumulative = [1.0]
    positions_hist = []
    current_weights: Dict[str, float] = {}
    tc_bps = transaction_cost_bps / 10000.0

    for i, date in enumerate(dates):
        rets = ret_by_date.get(date, {})

        # Daily return from existing positions
        daily_ret = sum(current_weights.get(s, 0) * rets.get(s, 0) for s in current_weights)

        # Rebalance
        if i % rebalance_freq == 0:
            sigs = sig_by_date.get(date, {})
            if len(sigs) >= top_n:
                ranked = sorted(sigs.items(), key=lambda x: x[1], reverse=True)
                new_weights: Dict[str, float] = {}

                # Long top_n
                long_weight = 1.0 / top_n if long_only else 0.5 / top_n
                for sym, _ in ranked[:top_n]:
                    new_weights[sym] = long_weight

                # Short bottom_n (if not long-only)
                if not long_only:
                    short_weight = -0.5 / top_n
                    for sym, _ in ranked[-top_n:]:
                        new_weights[sym] = new_weights.get(sym, 0) + short_weight

                # Transaction costs
                turnover = sum(
                    abs(new_weights.get(s, 0) - current_weights.get(s, 0))
                    for s in set(list(new_weights.keys()) + list(current_weights.keys()))
                )
                daily_ret -= turnover * tc_bps

                current_weights = new_weights

        daily_returns.append(daily_ret)
        cumulative.append(cumulative[-1] * (1 + daily_ret))
        positions_hist.append(dict(current_weights))

    # Performance metrics
    n = len(daily_returns)
    if n == 0:
        return _empty_portfolio_result()

    total_return = cumulative[-1] / cumulative[0] - 1
    ann_return = (1 + total_return) ** (252 / max(n, 1)) - 1
    mean_daily = sum(daily_returns) / n
    var_daily = sum((r - mean_daily) ** 2 for r in daily_returns) / max(n - 1, 1)
    ann_vol = math.sqrt(var_daily * 252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    peak = cumulative[0]
    max_dd = 0.0
    for v in cumulative:
        peak = max(peak, v)
        dd = (v - peak) / peak
        max_dd = min(max_dd, dd)

    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0

    # VaR/CVaR
    sorted_rets = sorted(daily_returns)
    var_idx = max(0, int(0.05 * n))
    var_95 = sorted_rets[var_idx]
    cvar_95 = sum(sorted_rets[:var_idx + 1]) / max(var_idx + 1, 1)

    # Higher moments
    skew = _skewness(daily_returns)
    kurt = _kurtosis(daily_returns)

    # Average turnover
    turnovers = []
    for i in range(1, len(positions_hist)):
        prev = positions_hist[i - 1]
        curr = positions_hist[i]
        all_syms = set(list(prev.keys()) + list(curr.keys()))
        to = sum(abs(curr.get(s, 0) - prev.get(s, 0)) for s in all_syms)
        turnovers.append(to)
    avg_turnover = sum(turnovers) / len(turnovers) if turnovers else 0

    return PortfolioBacktestResult(
        dates=dates,
        daily_returns=daily_returns,
        cumulative_returns=[c / cumulative[0] for c in cumulative[1:]],
        positions=positions_hist,
        total_return=total_return,
        annualized_return=ann_return,
        annualized_vol=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        avg_turnover=avg_turnover,
        var_95=var_95,
        cvar_95=cvar_95,
        worst_day=sorted_rets[0] if sorted_rets else 0,
        best_day=sorted_rets[-1] if sorted_rets else 0,
        skewness=skew,
        kurtosis=kurt,
    )


def _skewness(values: List[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    m = sum(values) / n
    m3 = sum((v - m) ** 3 for v in values) / n
    m2 = sum((v - m) ** 2 for v in values) / n
    if m2 <= 0:
        return 0.0
    return m3 / (m2 ** 1.5)


def _kurtosis(values: List[float]) -> float:
    n = len(values)
    if n < 4:
        return 0.0
    m = sum(values) / n
    m4 = sum((v - m) ** 4 for v in values) / n
    m2 = sum((v - m) ** 2 for v in values) / n
    if m2 <= 0:
        return 0.0
    return m4 / (m2 ** 2) - 3.0  # excess kurtosis


def _empty_portfolio_result() -> PortfolioBacktestResult:
    return PortfolioBacktestResult(
        dates=[], daily_returns=[], cumulative_returns=[], positions=[],
        total_return=0, annualized_return=0, annualized_vol=0,
        sharpe_ratio=0, max_drawdown=0, calmar_ratio=0, avg_turnover=0,
        var_95=0, cvar_95=0, worst_day=0, best_day=0, skewness=0, kurtosis=0,
    )
