"""Synthetic scenario engine — go beyond historical replay.

Three layers of scenario generation:

1. **Parametric stress tests** — Shock specific risk factors (rates, VIX, credit)
   and propagate through the covariance structure to all assets.

2. **Contagion scenarios** — Model cascading failures: one sector shock triggers
   cross-sector contagion with amplification and feedback loops.

3. **Hypothetical regime scenarios** — "What if we enter a 2022-style rate shock
   but with 2024-level AI concentration?" Blend historical regime dynamics
   with current portfolio composition.

The key insight vs. historical simulation: history is one sample path from
the distribution of possible outcomes. We need to explore the distribution,
not just replay the sample.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.simulation.covariance import CovEstimate, stress_cov, cholesky, nearest_psd
from src.simulation.generator import (
    SimulationResult,
    _correlated_innovations,
    simulate_factor_shock,
    simulate_scenario,
    summarize_simulation,
)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class ScenarioSpec:
    """Specification for a synthetic scenario."""
    name: str
    description: str
    # Factor shocks: factor_name → shock magnitude
    factor_shocks: Dict[str, float]
    # Asset-level target moves (optional)
    target_moves: Dict[str, float]
    # Regime parameters
    vol_multiplier: float = 1.0
    corr_multiplier: float = 1.0
    # Timing
    shock_horizon: int = 5     # days for shock to fully materialize
    decay_horizon: int = 20    # days for aftershocks
    # Contagion
    contagion_rounds: int = 0  # 0 = no contagion propagation


# Pre-built scenario templates
SCENARIO_TEMPLATES: Dict[str, ScenarioSpec] = {
    "rate_shock_100bp": ScenarioSpec(
        name="Rate Shock (+100bp)",
        description="10Y yield spikes 100bp over 5 days. Duration-sensitive assets sell off. "
                    "Credit spreads widen sympathetically. Flight to quality in short duration.",
        factor_shocks={"DGS10": 1.0, "DGS2": 0.5, "BAMLH0A0HYM2": 0.3},
        target_moves={
            "TLT": -0.12, "IEF": -0.06,           # bonds get crushed
            "XLU": -0.08, "XLRE": -0.10,           # rate-sensitive sectors
            "XLF": 0.03,                            # banks benefit
            "QQQ": -0.06, "SPY": -0.04,            # growth sells off
        },
        vol_multiplier=1.8, corr_multiplier=1.3,
        shock_horizon=5, decay_horizon=30,
    ),

    "credit_crisis": ScenarioSpec(
        name="Credit Crisis",
        description="HY spreads blow out 200bp. Leveraged names collapse. "
                    "Flight to treasuries. Equity vol spikes.",
        factor_shocks={"BAMLH0A0HYM2": 2.0, "VIXCLS": 15.0},
        target_moves={
            "HYG": -0.08, "JNK": -0.10,           # HY bonds
            "XLF": -0.15,                           # financials
            "SPY": -0.10, "IWM": -0.15,            # small caps hit harder
            "TLT": 0.05,                            # flight to quality
            "GLD": 0.04,                            # gold bid
        },
        vol_multiplier=2.5, corr_multiplier=1.6,
        shock_horizon=3, decay_horizon=40,
        contagion_rounds=2,
    ),

    "vix_spike": ScenarioSpec(
        name="VIX Spike to 40+",
        description="Sudden volatility event (flash crash, geopolitical shock). "
                    "VIX doubles in 2 days. All correlations spike. "
                    "Systematic deleveraging amplifies moves.",
        factor_shocks={"VIXCLS": 20.0},
        target_moves={
            "SPY": -0.08, "QQQ": -0.10, "IWM": -0.12,
            "UVXY": 0.60,                           # vol products explode
            "GLD": 0.03, "TLT": 0.04,              # safe havens
        },
        vol_multiplier=3.0, corr_multiplier=1.8,
        shock_horizon=2, decay_horizon=15,
    ),

    "tech_rotation": ScenarioSpec(
        name="Tech → Value Rotation",
        description="AI narrative cracks. Mega-cap tech sells off 15-20%. "
                    "Capital rotates to value, industrials, energy. "
                    "Broad index relatively contained due to rotation.",
        factor_shocks={},
        target_moves={
            "NVDA": -0.25, "META": -0.18, "MSFT": -0.12, "GOOGL": -0.15,
            "AAPL": -0.10, "AMZN": -0.12,
            "XLE": 0.08, "XLI": 0.06, "XLV": 0.04,  # value sectors rally
            "QQQ": -0.15, "SPY": -0.05,               # index impact muted
            "IWM": 0.05,                               # small caps benefit
        },
        vol_multiplier=1.5, corr_multiplier=0.8,  # lower corr (rotation, not panic)
        shock_horizon=10, decay_horizon=40,
    ),

    "stagflation": ScenarioSpec(
        name="Stagflation Scare",
        description="CPI reaccelerates + growth slows. Fed trapped. "
                    "Both bonds and stocks sell off. Commodities rally. "
                    "60/40 portfolio gets destroyed.",
        factor_shocks={"DGS10": 0.5, "DCOILWTICO": 15.0, "BAMLH0A0HYM2": 0.5},
        target_moves={
            "SPY": -0.08, "QQQ": -0.10, "TLT": -0.08,  # everything sells
            "XLE": 0.10, "GLD": 0.06,                     # inflation hedges
            "DBA": 0.08,                                   # commodities
            "XLP": -0.02, "XLU": -0.06,                   # even defensives hurt
        },
        vol_multiplier=1.6, corr_multiplier=1.4,
        shock_horizon=15, decay_horizon=60,
    ),

    "liquidity_crisis": ScenarioSpec(
        name="Liquidity Crisis",
        description="Repo/funding stress → forced selling across asset classes. "
                    "Bid-ask spreads blow out. Correlations go to 1. "
                    "Even 'safe' assets sell as margin calls cascade.",
        factor_shocks={"BAMLH0A0HYM2": 1.5, "VIXCLS": 25.0},
        target_moves={
            "SPY": -0.12, "QQQ": -0.15, "IWM": -0.18,
            "HYG": -0.10, "TLT": -0.03,  # even treasuries sell initially
            "GLD": -0.02,                  # gold sells for margin
        },
        vol_multiplier=3.0, corr_multiplier=2.0,  # correlations go extreme
        shock_horizon=3, decay_horizon=20,
        contagion_rounds=3,
    ),

    "geopolitical_shock": ScenarioSpec(
        name="Geopolitical Shock",
        description="Major geopolitical escalation. Oil spikes, safe havens bid. "
                    "Defense stocks rally. Supply chain disruption fears.",
        factor_shocks={"DCOILWTICO": 25.0, "VIXCLS": 10.0},
        target_moves={
            "SPY": -0.06, "QQQ": -0.07,
            "META": -0.05, "GOOGL": -0.06, "NVDA": -0.08,
            "XLE": 0.12,
            "LMT": 0.08, "RTX": 0.07, "NOC": 0.06,
            "GLD": 0.05, "TLT": 0.03,
            "XLI": -0.05,
        },
        vol_multiplier=1.8, corr_multiplier=1.2,
        shock_horizon=2, decay_horizon=30,
    ),

    "fed_pivot_dovish": ScenarioSpec(
        name="Fed Dovish Pivot",
        description="Surprise 50bp cut. Growth stocks rally hard. Bonds surge. "
                    "Dollar weakens. Full risk-on across the board.",
        factor_shocks={"DGS10": -0.75, "DGS2": -0.50},
        target_moves={
            "QQQ": 0.08, "SPY": 0.05, "IWM": 0.10,
            "NVDA": 0.12, "META": 0.08, "MSFT": 0.06, "GOOGL": 0.07,
            "AMZN": 0.07, "AAPL": 0.05, "TSLA": 0.10,
            "TLT": 0.08, "IEF": 0.04,
            "XLF": -0.03, "GLD": 0.04,
        },
        vol_multiplier=1.3, corr_multiplier=0.9,
        shock_horizon=3, decay_horizon=20,
    ),

    "ai_bubble_burst": ScenarioSpec(
        name="AI Bubble Burst",
        description="AI capex disappoints massively. Hyperscaler guidance slashed. "
                    "Mag-7 sell off 20-40%. Contagion to semis, cloud, data centers.",
        factor_shocks={"VIXCLS": 12.0},
        target_moves={
            "NVDA": -0.35, "META": -0.20, "MSFT": -0.18, "GOOGL": -0.22,
            "AMZN": -0.15, "AAPL": -0.10, "TSLA": -0.25,
            "QQQ": -0.20, "SPY": -0.10, "IWM": -0.05,
            "XLK": -0.18,
            "XLE": 0.03, "XLV": 0.02, "TLT": 0.03,
        },
        vol_multiplier=2.2, corr_multiplier=1.5,
        shock_horizon=5, decay_horizon=30,
        contagion_rounds=1,
    ),

    "china_contagion": ScenarioSpec(
        name="China Financial Contagion",
        description="Chinese property crisis escalates. Capital flight. "
                    "EM currencies crash. Commodity demand collapses.",
        factor_shocks={"DCOILWTICO": -15.0, "VIXCLS": 8.0},
        target_moves={
            "SPY": -0.06, "QQQ": -0.05,
            "AAPL": -0.10, "TSLA": -0.12,
            "META": -0.04, "GOOGL": -0.05, "NVDA": -0.08,
            "XLE": -0.08, "USO": -0.12,
            "GLD": 0.04, "TLT": 0.05,
        },
        vol_multiplier=1.8, corr_multiplier=1.4,
        shock_horizon=5, decay_horizon=40,
        contagion_rounds=1,
    ),

    "dollar_crisis": ScenarioSpec(
        name="Dollar Crisis",
        description="USD index drops 10%+. Treasury auction fails. "
                    "Foreign holders dump USTs. Gold and commodities surge.",
        factor_shocks={"DGS10": 0.8},
        target_moves={
            "TLT": -0.10, "IEF": -0.05,
            "GLD": 0.15, "USO": 0.10,
            "SPY": -0.05, "QQQ": -0.04,
            "META": -0.04, "GOOGL": -0.04, "NVDA": -0.05,
            "XLE": 0.06, "XLP": 0.02,
        },
        vol_multiplier=2.0, corr_multiplier=1.5,
        shock_horizon=5, decay_horizon=30,
    ),

    "bond_dislocation": ScenarioSpec(
        name="Treasury Dislocation",
        description="Basis trade unwind. Treasury liquidity evaporates. "
                    "Repo rates spike. SVB-style contagion but broader.",
        factor_shocks={"BAMLH0A0HYM2": 1.0, "DGS10": 0.6},
        target_moves={
            "TLT": -0.08, "IEF": -0.04,
            "HYG": -0.06, "LQD": -0.04,
            "XLF": -0.12,
            "SPY": -0.07, "IWM": -0.10,
            "META": -0.06, "GOOGL": -0.06, "NVDA": -0.08,
            "GLD": 0.03,
        },
        vol_multiplier=2.5, corr_multiplier=1.7,
        shock_horizon=3, decay_horizon=25,
        contagion_rounds=2,
    ),

    "pandemic_resurgence": ScenarioSpec(
        name="Pandemic Resurgence",
        description="New pathogen triggers lockdown fears. Travel and leisure collapse. "
                    "WFH stocks partially hedge. Bonds bid on growth fears.",
        factor_shocks={"VIXCLS": 15.0},
        target_moves={
            "SPY": -0.10, "IWM": -0.15,
            "XLI": -0.12, "XLY": -0.14,
            "META": -0.05, "GOOGL": -0.04, "MSFT": 0.02,
            "NVDA": -0.06, "AAPL": -0.08,
            "XLV": 0.05, "TLT": 0.06, "GLD": 0.04,
        },
        vol_multiplier=2.8, corr_multiplier=1.6,
        shock_horizon=3, decay_horizon=20,
    ),

    "inflation_reaccel": ScenarioSpec(
        name="Inflation Re-acceleration",
        description="CPI prints 6%+. Fed forced to hike again. "
                    "Both stocks and bonds sell off. 2022 playbook returns.",
        factor_shocks={"DGS10": 1.2, "DCOILWTICO": 20.0},
        target_moves={
            "SPY": -0.08, "QQQ": -0.12, "IWM": -0.10,
            "META": -0.10, "GOOGL": -0.08, "NVDA": -0.14,
            "MSFT": -0.08, "AAPL": -0.06, "TSLA": -0.15,
            "TLT": -0.12, "IEF": -0.06, "HYG": -0.05,
            "XLE": 0.08, "GLD": 0.05, "XLU": -0.08,
        },
        vol_multiplier=1.8, corr_multiplier=1.4,
        shock_horizon=10, decay_horizon=40,
    ),

    "tariff_war": ScenarioSpec(
        name="Tariff War Escalation",
        description="Broad 25% tariffs on all imports. Retaliation from partners. "
                    "Supply chains disrupted. Consumer prices surge. Growth slows.",
        factor_shocks={"DCOILWTICO": 10.0, "VIXCLS": 8.0},
        target_moves={
            "SPY": -0.08, "QQQ": -0.10, "IWM": -0.06,
            "AAPL": -0.15, "NVDA": -0.12, "TSLA": -0.14,
            "META": -0.06, "GOOGL": -0.07, "AMZN": -0.10,
            "XLI": -0.08, "XLP": -0.04,
            "GLD": 0.05, "TLT": 0.02,
        },
        vol_multiplier=1.6, corr_multiplier=1.3,
        shock_horizon=5, decay_horizon=40,
    ),
}


# ---------------------------------------------------------------------------
# Contagion model
# ---------------------------------------------------------------------------

def _propagate_contagion(
    target_moves: Dict[str, float],
    cov: CovEstimate,
    amplification: float = 0.5,
    threshold: float = 0.02,
) -> Dict[str, float]:
    """Propagate shocks through covariance structure with amplification.

    For each shocked asset, compute conditional impact on all other assets.
    Then iterate: newly-shocked assets propagate to their neighbors.
    Amplification < 1.0 ensures convergence (each round is weaker).

    Returns updated target_moves with contagion effects.
    """
    symbols = cov.symbols
    result = dict(target_moves)
    n = len(symbols)
    sym_to_idx = {s: i for i, s in enumerate(symbols)}

    # Assets already shocked
    shocked = {s for s in target_moves if s in sym_to_idx}
    new_shocks = dict(target_moves)

    for _ in range(3):  # max contagion rounds
        propagated: Dict[str, float] = {}
        for shock_sym, shock_mag in new_shocks.items():
            if shock_sym not in sym_to_idx:
                continue
            i = sym_to_idx[shock_sym]
            var_i = cov.cov[i][i]
            if var_i <= 0:
                continue

            for j in range(n):
                target_sym = symbols[j]
                if target_sym in shocked:
                    continue
                # Conditional expectation: cov(j,i)/var(i) * shock
                impact = cov.cov[j][i] / var_i * shock_mag * amplification
                if abs(impact) > threshold:
                    current = propagated.get(target_sym, 0.0)
                    propagated[target_sym] = current + impact

        if not propagated:
            break

        for sym, impact in propagated.items():
            result[sym] = result.get(sym, 0.0) + impact
            shocked.add(sym)

        new_shocks = propagated
        amplification *= 0.5  # each round weaker

    return result


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(
    spec: ScenarioSpec,
    base_cov: CovEstimate,
    start_prices: List[float],
    symbols: List[str],
    n_paths: int = 500,
    df: Optional[float] = 5.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a complete scenario simulation.

    1. Apply contagion propagation if specified
    2. Create stressed covariance
    3. Run conditional simulation
    4. Compute summary statistics

    Returns full results including per-asset distribution of outcomes.
    """
    target_moves = dict(spec.target_moves)

    # Contagion propagation
    if spec.contagion_rounds > 0:
        for _ in range(spec.contagion_rounds):
            target_moves = _propagate_contagion(
                target_moves, base_cov, amplification=0.5,
            )

    # Create stressed covariance
    stressed = stress_cov(
        base_cov,
        correlation_multiplier=spec.corr_multiplier,
        vol_multiplier=spec.vol_multiplier,
    )

    # Run simulation
    sim = simulate_scenario(
        base_cov=stressed,
        start_prices=start_prices,
        symbols=symbols,
        target_moves=target_moves,
        move_horizon=spec.shock_horizon + spec.decay_horizon,
        n_paths=n_paths,
        df=df,
        seed=seed,
    )

    summary = summarize_simulation(sim)

    return {
        "scenario": spec.name,
        "description": spec.description,
        "target_moves": target_moves,
        "stressed_vols": {s: v for s, v in zip(stressed.symbols, stressed.vols)},
        "summary": summary,
        "simulation": sim,
    }


def run_all_scenarios(
    base_cov: CovEstimate,
    start_prices: List[float],
    symbols: List[str],
    scenarios: Optional[List[str]] = None,
    n_paths: int = 500,
    df: Optional[float] = 5.0,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Run multiple scenario simulations.

    Args:
        scenarios: list of scenario template names, or None for all
    """
    if scenarios is None:
        scenarios = list(SCENARIO_TEMPLATES.keys())

    results = {}
    for name in scenarios:
        spec = SCENARIO_TEMPLATES.get(name)
        if spec is None:
            continue
        results[name] = run_scenario(
            spec=spec,
            base_cov=base_cov,
            start_prices=start_prices,
            symbols=symbols,
            n_paths=n_paths,
            df=df,
            seed=seed,
        )

    return results


# ---------------------------------------------------------------------------
# Custom scenario builder
# ---------------------------------------------------------------------------

def build_custom_scenario(
    name: str,
    description: str,
    asset_shocks: Dict[str, float],
    vol_regime: str = "elevated",
    contagion: bool = False,
) -> ScenarioSpec:
    """Build a custom scenario from user-specified parameters.

    vol_regime: "normal", "elevated", "crisis"
    """
    vol_map = {"normal": 1.0, "elevated": 1.5, "crisis": 2.5}
    corr_map = {"normal": 1.0, "elevated": 1.3, "crisis": 1.8}

    return ScenarioSpec(
        name=name,
        description=description,
        factor_shocks={},
        target_moves=asset_shocks,
        vol_multiplier=vol_map.get(vol_regime, 1.5),
        corr_multiplier=corr_map.get(vol_regime, 1.3),
        shock_horizon=5,
        decay_horizon=20,
        contagion_rounds=2 if contagion else 0,
    )
