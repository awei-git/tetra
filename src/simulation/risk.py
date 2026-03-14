"""Risk decomposition and measurement.

Answers the questions a portfolio manager actually needs:

1. How much can I lose? → VaR, CVaR (parametric + simulation-based)
2. Where is the risk coming from? → Factor attribution, marginal contributions
3. What's my concentration risk? → HHI, effective N, sector exposure
4. How does risk change under stress? → Conditional risk metrics
5. Should I be scared right now? → Regime-aware risk scaling

Two approaches:
- **Parametric**: Fast, based on covariance matrix. Good for daily monitoring.
- **Simulation-based**: Uses the full simulation engine. Better for tail risk
  and non-linear exposures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.simulation.covariance import CovEstimate, annualize_vol
from src.simulation.generator import SimulationResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PortfolioRisk:
    """Complete risk profile for a portfolio."""
    # Portfolio-level
    total_vol_ann: float          # annualized portfolio volatility
    var_95_1d: float              # 1-day 95% VaR (dollar or %)
    var_99_1d: float              # 1-day 99% VaR
    cvar_95_1d: float             # 1-day 95% CVaR (expected shortfall)
    cvar_99_1d: float
    # Drawdown risk
    expected_max_drawdown_1y: float  # estimated max drawdown over 1 year
    # Concentration
    hhi: float                    # Herfindahl-Hirschman Index
    effective_n: float            # effective number of positions (1/HHI)
    top_3_weight: float           # weight of top 3 positions
    # Factor contributions
    marginal_risk: Dict[str, float]    # symbol → marginal risk contribution
    component_risk: Dict[str, float]   # symbol → component risk (sums to total)
    # Sector exposure (if available)
    sector_risk: Dict[str, float]      # sector → risk contribution


@dataclass(frozen=True)
class StressTestResult:
    """Result of a stress test on the portfolio."""
    scenario_name: str
    portfolio_pnl: float          # expected P&L
    portfolio_pnl_pct: float      # as % of portfolio value
    var_95_under_stress: float    # VaR under stressed covariance
    worst_position: str           # worst-performing position
    worst_position_pnl: float
    position_pnls: Dict[str, float]


# ---------------------------------------------------------------------------
# Parametric risk (covariance-based)
# ---------------------------------------------------------------------------

def parametric_risk(
    weights: Dict[str, float],
    cov: CovEstimate,
    portfolio_value: float = 1.0,
    confidence_levels: Sequence[float] = (0.95, 0.99),
) -> PortfolioRisk:
    """Compute parametric risk metrics from covariance matrix.

    Args:
        weights: symbol → portfolio weight (should sum to ~1.0 for fully invested)
        cov: covariance estimate (daily returns scale)
        portfolio_value: total portfolio value for dollar VaR
        confidence_levels: VaR confidence levels
    """
    symbols = cov.symbols
    n = len(symbols)
    w = [weights.get(s, 0.0) for s in symbols]

    # Portfolio variance: w' * Σ * w
    port_var = 0.0
    for i in range(n):
        for j in range(n):
            port_var += w[i] * w[j] * cov.cov[i][j]

    port_vol_daily = math.sqrt(max(port_var, 0.0))
    port_vol_ann = annualize_vol(port_vol_daily)

    # Parametric VaR (assumes normality — conservative for fat tails)
    z_95 = 1.645
    z_99 = 2.326
    var_95 = port_vol_daily * z_95 * portfolio_value
    var_99 = port_vol_daily * z_99 * portfolio_value

    # CVaR (expected shortfall under normality)
    # E[X | X > VaR] = μ + σ * φ(z) / (1-Φ(z))
    phi_95 = _normal_pdf(z_95)
    phi_99 = _normal_pdf(z_99)
    cvar_95 = port_vol_daily * phi_95 / 0.05 * portfolio_value
    cvar_99 = port_vol_daily * phi_99 / 0.01 * portfolio_value

    # Expected max drawdown (Magdon-Ismail approximation)
    # E[MDD] ≈ sqrt(π/2) * σ * sqrt(T) for geometric Brownian motion
    # For 252 trading days:
    expected_mdd = math.sqrt(math.pi / 2) * port_vol_daily * math.sqrt(252)

    # Concentration
    abs_weights = [abs(wi) for wi in w if wi != 0]
    total_abs = sum(abs_weights)
    if total_abs > 0:
        norm_weights = [wi / total_abs for wi in abs_weights]
        hhi = sum(wi ** 2 for wi in norm_weights)
        effective_n = 1.0 / hhi if hhi > 0 else 0
    else:
        hhi = 1.0
        effective_n = 0.0

    # Top 3 weight
    sorted_abs = sorted(abs_weights, reverse=True)
    top_3 = sum(sorted_abs[:3]) / total_abs if total_abs > 0 else 0

    # Marginal risk contribution: ∂σ_p/∂w_i = (Σ * w)_i / σ_p
    marginal = {}
    component = {}
    cov_w = [sum(cov.cov[i][j] * w[j] for j in range(n)) for i in range(n)]
    for i in range(n):
        if port_vol_daily > 0:
            mr = cov_w[i] / port_vol_daily
            marginal[symbols[i]] = annualize_vol(mr)
            # Component risk: w_i * marginal_i (sums to total vol)
            component[symbols[i]] = w[i] * annualize_vol(mr)
        else:
            marginal[symbols[i]] = 0.0
            component[symbols[i]] = 0.0

    return PortfolioRisk(
        total_vol_ann=port_vol_ann,
        var_95_1d=var_95,
        var_99_1d=var_99,
        cvar_95_1d=cvar_95,
        cvar_99_1d=cvar_99,
        expected_max_drawdown_1y=expected_mdd,
        hhi=hhi,
        effective_n=effective_n,
        top_3_weight=top_3,
        marginal_risk=marginal,
        component_risk=component,
        sector_risk={},
    )


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


# ---------------------------------------------------------------------------
# Simulation-based risk (uses SimulationResult from generator)
# ---------------------------------------------------------------------------

def simulation_risk(
    sim: SimulationResult,
    weights: Dict[str, float],
    portfolio_value: float = 1.0,
) -> PortfolioRisk:
    """Compute risk metrics from Monte Carlo simulation.

    Better than parametric for:
    - Fat tails (Student-t innovations)
    - Regime switches (non-stationary dynamics)
    - Non-linear exposures (options, convexity)
    """
    symbols = sim.symbols
    n = len(symbols)
    w = [weights.get(s, 0.0) for s in symbols]

    # Portfolio daily returns across all paths
    all_daily = []
    for p in range(sim.n_paths):
        for t in range(sim.horizon):
            daily_port = sum(
                w[j] * sim.returns[p][t][j] for j in range(n)
            )
            all_daily.append(daily_port)

    # 1-day VaR and CVaR from empirical daily return distribution
    all_daily_sorted = sorted(all_daily)
    n_daily = len(all_daily_sorted)
    idx_95 = max(0, int(0.05 * n_daily))
    idx_99 = max(0, int(0.01 * n_daily))
    var_95 = -all_daily_sorted[idx_95] * portfolio_value
    var_99 = -all_daily_sorted[idx_99] * portfolio_value
    cvar_95 = -sum(all_daily_sorted[:idx_95 + 1]) / max(idx_95 + 1, 1) * portfolio_value
    cvar_99 = -sum(all_daily_sorted[:idx_99 + 1]) / max(idx_99 + 1, 1) * portfolio_value

    mean_daily = sum(all_daily) / len(all_daily) if all_daily else 0
    var_daily = sum((r - mean_daily) ** 2 for r in all_daily) / max(len(all_daily) - 1, 1) if all_daily else 0
    port_vol_ann = annualize_vol(math.sqrt(var_daily))

    # Max drawdown distribution
    drawdowns = []
    for p in range(sim.n_paths):
        peak = portfolio_value
        max_dd = 0.0
        value = portfolio_value
        for t in range(sim.horizon):
            daily_ret = sum(w[j] * sim.returns[p][t][j] for j in range(n))
            value *= (1 + daily_ret)
            peak = max(peak, value)
            dd = (value - peak) / peak
            max_dd = min(max_dd, dd)
        drawdowns.append(max_dd)
    expected_mdd = sum(drawdowns) / len(drawdowns) if drawdowns else 0

    # Concentration (same as parametric)
    abs_weights = [abs(wi) for wi in w if wi != 0]
    total_abs = sum(abs_weights)
    if total_abs > 0:
        norm_w = [wi / total_abs for wi in abs_weights]
        hhi = sum(wi ** 2 for wi in norm_w)
        effective_n = 1.0 / hhi if hhi > 0 else 0
    else:
        hhi = 1.0
        effective_n = 0

    sorted_abs = sorted(abs_weights, reverse=True)
    top_3 = sum(sorted_abs[:3]) / total_abs if total_abs > 0 else 0

    # Marginal risk: numerical approximation via daily returns
    # Bump each weight by epsilon, recompute daily portfolio VaR
    epsilon = 0.01
    marginal = {}
    component = {}
    daily_idx_95 = max(0, int(0.05 * n_daily))
    for j in range(n):
        bumped_w = w[:]
        bumped_w[j] += epsilon
        bumped_daily = []
        for p in range(sim.n_paths):
            for t in range(sim.horizon):
                bumped_daily.append(sum(bumped_w[i] * sim.returns[p][t][i] for i in range(n)))
        bumped_daily.sort()
        bumped_var = -bumped_daily[daily_idx_95] * portfolio_value
        marginal_var = (bumped_var - var_95) / epsilon
        marginal[symbols[j]] = marginal_var
        component[symbols[j]] = w[j] * marginal_var

    return PortfolioRisk(
        total_vol_ann=port_vol_ann,
        var_95_1d=var_95,
        var_99_1d=var_99,
        cvar_95_1d=cvar_95,
        cvar_99_1d=cvar_99,
        expected_max_drawdown_1y=expected_mdd,
        hhi=hhi,
        effective_n=effective_n,
        top_3_weight=top_3,
        marginal_risk=marginal,
        component_risk=component,
        sector_risk={},
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

def stress_test_portfolio(
    weights: Dict[str, float],
    scenario_results: Dict[str, Dict],
    portfolio_value: float = 1.0,
) -> List[StressTestResult]:
    """Run portfolio through pre-computed scenario results.

    scenario_results: output from scenarios.run_all_scenarios()
    """
    results = []
    symbols = list(weights.keys())

    for name, scenario_data in scenario_results.items():
        target_moves = scenario_data.get("target_moves", {})

        # Compute portfolio P&L under scenario
        position_pnls = {}
        total_pnl = 0.0
        worst_sym = ""
        worst_pnl = 0.0

        # Build lookup from simulation summary for fallback
        sim_moves: Dict[str, float] = {}
        summary = scenario_data.get("summary", {})
        for asset_stat in summary.get("assets", []):
            sim_moves[asset_stat.get("symbol", "")] = asset_stat.get("mean_return", 0.0)

        for sym, w in weights.items():
            # Prefer explicit target_moves; fall back to simulated mean return
            asset_move = target_moves.get(sym, sim_moves.get(sym, 0.0))
            pnl = w * asset_move * portfolio_value
            position_pnls[sym] = pnl
            total_pnl += pnl
            if pnl < worst_pnl:
                worst_pnl = pnl
                worst_sym = sym

        # VaR under stress from simulation summary
        summary = scenario_data.get("summary", {})
        assets = summary.get("assets", [])
        stress_var = 0.0
        for asset_stat in assets:
            sym = asset_stat.get("symbol", "")
            w = weights.get(sym, 0.0)
            p05 = asset_stat.get("p05", 0.0)
            stress_var += abs(w * p05) * portfolio_value

        results.append(StressTestResult(
            scenario_name=name,
            portfolio_pnl=total_pnl,
            portfolio_pnl_pct=total_pnl / portfolio_value if portfolio_value > 0 else 0,
            var_95_under_stress=stress_var,
            worst_position=worst_sym,
            worst_position_pnl=worst_pnl,
            position_pnls=position_pnls,
        ))

    # Sort by worst outcome
    results.sort(key=lambda r: r.portfolio_pnl)
    return results


# ---------------------------------------------------------------------------
# Risk attribution helpers
# ---------------------------------------------------------------------------

def risk_budget_check(
    risk: PortfolioRisk,
    max_vol: float = 0.20,
    max_single_name_pct: float = 0.20,
    max_drawdown: float = 0.15,
    max_var_pct: float = 0.03,
) -> Dict[str, Any]:
    """Check portfolio risk against predefined budgets.

    Returns dict of budget name → {limit, actual, breach: bool, severity}.
    """
    checks = {}

    checks["annualized_vol"] = {
        "limit": max_vol,
        "actual": risk.total_vol_ann,
        "breach": risk.total_vol_ann > max_vol,
        "severity": (risk.total_vol_ann - max_vol) / max_vol if risk.total_vol_ann > max_vol else 0,
    }

    # Check single-name concentration
    if risk.component_risk:
        total_risk = sum(abs(v) for v in risk.component_risk.values())
        for sym, cr in risk.component_risk.items():
            pct = abs(cr) / total_risk if total_risk > 0 else 0
            if pct > max_single_name_pct:
                checks[f"concentration_{sym}"] = {
                    "limit": max_single_name_pct,
                    "actual": pct,
                    "breach": True,
                    "severity": (pct - max_single_name_pct) / max_single_name_pct,
                }

    checks["expected_max_drawdown"] = {
        "limit": max_drawdown,
        "actual": abs(risk.expected_max_drawdown_1y),
        "breach": abs(risk.expected_max_drawdown_1y) > max_drawdown,
        "severity": (abs(risk.expected_max_drawdown_1y) - max_drawdown) / max_drawdown
                    if abs(risk.expected_max_drawdown_1y) > max_drawdown else 0,
    }

    return checks


def position_sizing_from_risk(
    target_vol: float,
    cov: CovEstimate,
    signal_scores: Dict[str, float],
    max_weight: float = 0.10,
    min_weight: float = 0.0,
) -> Dict[str, float]:
    """Size positions to hit target portfolio volatility.

    Uses inverse-volatility weighting scaled by signal strength:
    - Higher signal → larger position (direction)
    - Higher vol → smaller position (risk budget)
    - Total portfolio vol targets `target_vol`

    Args:
        target_vol: target annualized portfolio volatility
        cov: covariance estimate
        signal_scores: symbol → signal score (-1 to 1)
        max_weight: maximum absolute weight per position
        min_weight: minimum absolute weight (0 = allow zero)
    """
    symbols = cov.symbols
    n = len(symbols)

    # Inverse-vol weights, scaled by signal
    raw_weights = {}
    for i, sym in enumerate(symbols):
        vol = cov.vols[i]  # annualized
        signal = signal_scores.get(sym, 0.0)
        if vol > 0 and abs(signal) > 0.01:
            raw_weights[sym] = signal / vol
        else:
            raw_weights[sym] = 0.0

    # Normalize to sum of abs = 1
    total_abs = sum(abs(v) for v in raw_weights.values())
    if total_abs == 0:
        return {s: 0.0 for s in symbols}

    weights = {s: v / total_abs for s, v in raw_weights.items()}

    # Apply position limits
    for s in weights:
        if abs(weights[s]) > max_weight:
            weights[s] = max_weight * (1 if weights[s] > 0 else -1)
        if abs(weights[s]) < min_weight:
            weights[s] = 0.0

    # Scale to hit target vol
    # Portfolio vol = sqrt(w' Σ w)
    w_vec = [weights.get(s, 0) for s in symbols]
    port_var = sum(
        w_vec[i] * w_vec[j] * cov.cov[i][j]
        for i in range(n) for j in range(n)
    )
    port_vol = annualize_vol(math.sqrt(max(port_var, 0)))

    if port_vol > 0:
        scale = target_vol / port_vol
        weights = {s: v * scale for s, v in weights.items()}

        # Re-apply limits after scaling
        for s in weights:
            if abs(weights[s]) > max_weight:
                weights[s] = max_weight * (1 if weights[s] > 0 else -1)

    return weights
