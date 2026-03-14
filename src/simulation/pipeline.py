"""Simulation pipeline — connects DB data to the simulation engine.

Loads market data from PostgreSQL, runs the full simulation stack:
1. Covariance estimation (Ledoit-Wolf + EWMA)
2. HMM regime detection
3. Regime-conditional covariance
4. Multi-asset simulation (regime-switching + fat tails)
5. Scenario stress tests (7 built-in + custom)
6. Risk decomposition (parametric + simulation-based)
7. Signal validation (walk-forward IC)

Results are stored back to simulation.* schema tables.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.schema import (
    simulation_covariance,
    simulation_regimes,
    simulation_risk,
    simulation_scenarios,
)
from src.db.session import engine
from src.simulation.covariance import (
    CovEstimate,
    ewma_cov,
    ledoit_wolf,
    regime_conditional_cov,
    stress_cov,
)
from src.simulation.regime import (
    RegimeModel,
    current_regime,
    extract_regime_features,
    fit_hmm,
    regime_transition_forecast,
)
from src.simulation.generator import (
    simulate_regime_switching,
    summarize_simulation,
)
from src.simulation.scenarios import (
    SCENARIO_TEMPLATES,
    run_scenario,
)
from src.simulation.risk import (
    parametric_risk,
    position_sizing_from_risk,
    risk_budget_check,
    simulation_risk as compute_sim_risk,
    stress_test_portfolio,
)

logger = logging.getLogger(__name__)
UTC = timezone.utc

# Core symbols for covariance/regime estimation (liquid, representative)
CORE_SYMBOLS = [
    # Mega cap
    "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "TSLA",
    # Sector ETFs
    "XLK", "XLE", "XLF", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU",
    # Indices
    "SPY", "QQQ", "IWM",
    # Bonds & rates
    "TLT", "IEF", "HYG", "LQD",
    # Commodities & safe havens
    "GLD", "USO",
    # Volatility
    "UVXY",
    # International
    "EEM", "EFA",
    # Crypto proxy
    "IBIT",
]


# ---------------------------------------------------------------------------
# Data loading from DB
# ---------------------------------------------------------------------------

async def _load_returns_matrix(
    symbols: List[str],
    lookback_days: int = 504,
    as_of: Optional[date] = None,
) -> Tuple[List[str], List[List[float]], List[str]]:
    """Load daily close prices and compute log returns.

    Returns:
        (symbols_with_data, returns_matrix T x N, dates)
    """
    target = as_of or datetime.now(tz=UTC).date()
    start = target - timedelta(days=int(lookback_days * 1.5))  # buffer for weekends

    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT symbol, DATE(timestamp) AS day, close
            FROM market.ohlcv
            WHERE symbol = ANY(:symbols)
              AND timestamp >= :start
              AND timestamp <= :end
            ORDER BY day, symbol
        """), {
            "symbols": symbols,
            "start": datetime.combine(start, datetime.min.time(), tzinfo=UTC),
            "end": datetime.combine(target, datetime.max.time(), tzinfo=UTC),
        })
        rows = result.fetchall()

    if not rows:
        return [], [], []

    # Build price matrix: date → {symbol → close}
    price_by_date: Dict[date, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r.close is not None:
            price_by_date[r.day][r.symbol] = float(r.close)

    dates = sorted(price_by_date.keys())

    # Filter symbols that have data for >= 80% of dates
    min_dates = int(len(dates) * 0.8)
    valid_symbols = []
    for sym in symbols:
        count = sum(1 for d in dates if sym in price_by_date[d])
        if count >= min_dates:
            valid_symbols.append(sym)

    if not valid_symbols:
        return [], [], []

    # Compute log returns
    returns = []
    return_dates = []
    for i in range(1, len(dates)):
        row = []
        skip = False
        for sym in valid_symbols:
            prev = price_by_date[dates[i - 1]].get(sym)
            curr = price_by_date[dates[i]].get(sym)
            if prev is None or curr is None or prev <= 0 or curr <= 0:
                row.append(0.0)  # fill missing with 0
            else:
                row.append(math.log(curr / prev))
        returns.append(row)
        return_dates.append(dates[i].isoformat())

    # Trim to lookback
    if len(returns) > lookback_days:
        returns = returns[-lookback_days:]
        return_dates = return_dates[-lookback_days:]

    return valid_symbols, returns, return_dates


async def _load_portfolio_weights(as_of: Optional[date] = None) -> Tuple[Dict[str, float], float]:
    """Load current portfolio positions as weights.

    Returns (weights_dict, total_value).
    """
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT symbol, shares, current_price, market_value, weight
            FROM portfolio.positions
        """))
        positions = result.fetchall()

        result = await conn.execute(text("""
            SELECT total_value, cash FROM portfolio.snapshots
            ORDER BY date DESC LIMIT 1
        """))
        snap = result.fetchone()

    if not positions:
        return {}, 0.0

    total_value = float(snap.total_value) if snap else 0.0
    weights = {}
    for pos in positions:
        w = float(pos.weight) if pos.weight else 0.0
        weights[pos.symbol] = w

    return weights, total_value


async def _load_start_prices(
    symbols: List[str],
    as_of: Optional[date] = None,
) -> List[float]:
    """Load latest close prices for simulation starting points."""
    target = as_of or datetime.now(tz=UTC).date()

    async with engine.begin() as conn:
        result = await conn.execute(text("""
            WITH latest AS (
                SELECT symbol, close,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
                FROM market.ohlcv
                WHERE symbol = ANY(:symbols)
                  AND timestamp::date <= :date
            )
            SELECT symbol, close FROM latest WHERE rn = 1
        """), {"symbols": symbols, "date": target})
        price_map = {r.symbol: float(r.close) for r in result.fetchall()}

    return [price_map.get(s, 100.0) for s in symbols]


# ---------------------------------------------------------------------------
# Store results to DB
# ---------------------------------------------------------------------------

async def _store_regime(as_of: date, model: RegimeModel) -> None:
    """Store regime detection results."""
    state, probs = current_regime(model)
    forecast = regime_transition_forecast(model, horizon=5)

    row = {
        "as_of": as_of,
        "n_states": model.n_states,
        "current_regime": state.label,
        "current_probs": json.dumps({
            model.states[i].label: round(probs[i], 4)
            for i in range(model.n_states)
        }),
        "transition_matrix": json.dumps([
            [round(v, 4) for v in row]
            for row in model.transition_matrix
        ]),
        "regime_states": json.dumps([
            {
                "label": s.label,
                "mean_vol": round(s.mean_vol, 4),
                "mean_return": round(s.mean_return, 6),
                "mean_corr": round(s.mean_corr, 4),
                "duration_days": round(s.duration_days, 1),
                "frequency": round(s.frequency, 4),
            }
            for s in model.states
        ]),
        "regime_forecast_5d": json.dumps([
            {model.states[i].label: round(forecast[d][i], 4) for i in range(model.n_states)}
            for d in range(len(forecast))
        ]),
        "log_likelihood": round(model.log_likelihood, 4),
        "n_observations": len(model.regime_path),
    }

    async with engine.begin() as conn:
        stmt = pg_insert(simulation_regimes).values(row)
        await conn.execute(stmt.on_conflict_do_update(
            index_elements=[simulation_regimes.c.as_of],
            set_={k: stmt.excluded[k] for k in row if k != "as_of"},
        ))


async def _store_covariance(as_of: date, cov: CovEstimate) -> None:
    """Store covariance estimate."""
    row = {
        "as_of": as_of,
        "method": cov.method.split("(")[0],  # just the method name
        "symbols": cov.symbols,
        "vols_ann": json.dumps({s: round(v, 6) for s, v in zip(cov.symbols, cov.vols)}),
        "correlation_matrix": json.dumps([
            [round(cov.corr[i][j], 4) for j in range(len(cov.symbols))]
            for i in range(len(cov.symbols))
        ]),
        "n_observations": cov.n_obs,
        "effective_observations": round(cov.effective_obs, 2),
    }

    async with engine.begin() as conn:
        stmt = pg_insert(simulation_covariance).values(row)
        await conn.execute(stmt.on_conflict_do_update(
            index_elements=[simulation_covariance.c.as_of, simulation_covariance.c.method],
            set_={k: stmt.excluded[k] for k in row if k not in ("as_of", "method")},
        ))


async def _store_risk(as_of: date, risk, method: str, breaches: Dict) -> None:
    """Store risk metrics."""
    row = {
        "as_of": as_of,
        "method": method,
        "total_vol_ann": round(risk.total_vol_ann, 6),
        "var_95_1d": round(risk.var_95_1d, 4),
        "var_99_1d": round(risk.var_99_1d, 4),
        "cvar_95_1d": round(risk.cvar_95_1d, 4),
        "cvar_99_1d": round(risk.cvar_99_1d, 4),
        "expected_max_drawdown": round(risk.expected_max_drawdown_1y, 6),
        "hhi": round(risk.hhi, 6),
        "effective_n": round(risk.effective_n, 4),
        "marginal_risk": json.dumps({k: round(v, 6) for k, v in risk.marginal_risk.items()}),
        "component_risk": json.dumps({k: round(v, 6) for k, v in risk.component_risk.items()}),
        "risk_budget_breaches": json.dumps(breaches),
    }

    async with engine.begin() as conn:
        stmt = pg_insert(simulation_risk).values(row)
        await conn.execute(stmt.on_conflict_do_update(
            index_elements=[simulation_risk.c.as_of, simulation_risk.c.method],
            set_={k: stmt.excluded[k] for k in row if k not in ("as_of", "method")},
        ))


async def _store_scenario(as_of: date, name: str, stress_result, summary: Dict) -> None:
    """Store scenario stress test result."""
    # Clamp extreme values to fit DB precision constraints
    _clamp_dollar = lambda v: max(-9.999e15, min(9.999e15, v))
    _clamp_pct = lambda v: max(-999.0, min(999.0, v))
    clamped_pnls = {k: round(_clamp_dollar(v), 2) for k, v in stress_result.position_pnls.items()}

    row = {
        "as_of": as_of,
        "scenario_name": name,
        "description": SCENARIO_TEMPLATES.get(name, type("", (), {"description": ""})()).description
                       if hasattr(SCENARIO_TEMPLATES.get(name), "description") else "",
        "portfolio_pnl": round(_clamp_dollar(stress_result.portfolio_pnl), 4),
        "portfolio_pnl_pct": round(_clamp_pct(stress_result.portfolio_pnl_pct), 6),
        "var_95_under_stress": round(_clamp_dollar(stress_result.var_95_under_stress), 4),
        "worst_position": stress_result.worst_position,
        "worst_position_pnl": round(_clamp_dollar(stress_result.worst_position_pnl), 4),
        "target_moves": json.dumps(
            {k: round(v, 4) for k, v in stress_result.position_pnls.items()}
        ),
        "position_pnls": json.dumps(clamped_pnls),
        "summary_stats": json.dumps(summary),
    }

    async with engine.begin() as conn:
        stmt = pg_insert(simulation_scenarios).values(row)
        await conn.execute(stmt.on_conflict_do_update(
            index_elements=[simulation_scenarios.c.as_of, simulation_scenarios.c.scenario_name],
            set_={k: stmt.excluded[k] for k in row if k not in ("as_of", "scenario_name")},
        ))


# ---------------------------------------------------------------------------
# Main pipeline stages
# ---------------------------------------------------------------------------

async def run_covariance_estimation(
    as_of: Optional[date] = None,
    lookback: int = 504,
) -> Dict[str, Any]:
    """Stage 1: Estimate covariance matrices."""
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    logger.info(f"Loading returns for {len(CORE_SYMBOLS)} symbols, lookback={lookback}d")
    symbols, returns, dates = await _load_returns_matrix(CORE_SYMBOLS, lookback, as_of)

    if not returns:
        return {"status": "no_data", "as_of": as_of.isoformat()}

    logger.info(f"Computing covariance: {len(symbols)} symbols, {len(returns)} observations")

    # Ledoit-Wolf (stable, for portfolio optimization)
    cov_lw = ledoit_wolf(returns, symbols)
    await _store_covariance(as_of, cov_lw)

    # EWMA (responsive, for short-horizon risk)
    cov_ewma = ewma_cov(returns, symbols, halflife=21)
    await _store_covariance(as_of, cov_ewma)

    logger.info(
        f"Covariance: LW method={cov_lw.method}, "
        f"top vols: {sorted(zip(symbols, cov_lw.vols), key=lambda x: -x[1])[:5]}"
    )

    return {
        "status": "ok",
        "as_of": as_of.isoformat(),
        "symbols": symbols,
        "n_obs": len(returns),
        "lw_method": cov_lw.method,
        # Pass through for downstream stages
        "_symbols": symbols,
        "_returns": returns,
        "_dates": dates,
        "_cov_lw": cov_lw,
        "_cov_ewma": cov_ewma,
    }


async def run_regime_detection(
    as_of: Optional[date] = None,
    cov_result: Optional[Dict] = None,
    lookback: int = 504,
) -> Dict[str, Any]:
    """Stage 2: Detect market regime via HMM."""
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    # Reuse data from covariance stage if available
    if cov_result and "_returns" in cov_result:
        symbols = cov_result["_symbols"]
        returns = cov_result["_returns"]
    else:
        symbols, returns, _ = await _load_returns_matrix(CORE_SYMBOLS, lookback, as_of)

    if not returns or len(returns) < 60:
        return {"status": "no_data", "as_of": as_of.isoformat()}

    logger.info(f"Fitting HMM on {len(returns)} observations, {len(symbols)} assets")

    # Extract features and fit HMM
    window = 20
    features = extract_regime_features(returns, window=window)
    model = fit_hmm(features, n_states=3, max_iter=100)

    state, probs = current_regime(model)
    forecast = regime_transition_forecast(model, horizon=5)

    await _store_regime(as_of, model)

    # Regime-conditional covariance
    aligned_returns = returns[window - 1:]
    regime_covs = regime_conditional_cov(aligned_returns, symbols, model.regime_path, shrink=True)

    logger.info(
        f"Regime: {state.label} (conf={max(probs):.2f}), "
        f"states: {[(s.label, f'{s.frequency:.0%}') for s in model.states]}"
    )

    return {
        "status": "ok",
        "as_of": as_of.isoformat(),
        "current_regime": state.label,
        "regime_probs": {model.states[i].label: round(probs[i], 4) for i in range(model.n_states)},
        "states": [
            {"label": s.label, "vol": f"{s.mean_vol:.1%}", "freq": f"{s.frequency:.0%}"}
            for s in model.states
        ],
        "forecast_5d": {
            model.states[i].label: round(forecast[4][i], 4)
            for i in range(model.n_states)
        },
        # Pass through
        "_model": model,
        "_regime_covs": regime_covs,
        "_symbols": symbols,
        "_returns": returns,
    }


async def run_risk_analysis(
    as_of: Optional[date] = None,
    cov_result: Optional[Dict] = None,
    regime_result: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Stage 3: Compute portfolio risk metrics."""
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    weights, total_value = await _load_portfolio_weights(as_of)
    if not weights or total_value <= 0:
        return {"status": "no_portfolio", "as_of": as_of.isoformat()}

    # Get covariance
    cov_lw = cov_result.get("_cov_lw") if cov_result else None
    if cov_lw is None:
        symbols, returns, _ = await _load_returns_matrix(CORE_SYMBOLS, 504, as_of)
        if returns:
            cov_lw = ledoit_wolf(returns, symbols)
        else:
            return {"status": "no_data", "as_of": as_of.isoformat()}

    # Parametric risk
    risk_param = parametric_risk(weights, cov_lw, portfolio_value=total_value)
    breaches_param = risk_budget_check(risk_param)
    breach_list = {k: v for k, v in breaches_param.items() if v.get("breach")}
    await _store_risk(as_of, risk_param, "parametric", breach_list)

    # Simulation-based risk (if regime model available)
    sim_risk_result = None
    if regime_result and "_model" in regime_result and "_regime_covs" in regime_result:
        model = regime_result["_model"]
        regime_covs = regime_result["_regime_covs"]
        symbols = regime_result["_symbols"]
        start_prices = await _load_start_prices(symbols, as_of)

        sim = simulate_regime_switching(
            regime_model=model,
            regime_covs=regime_covs,
            start_prices=start_prices,
            symbols=symbols,
            horizon=60,
            n_paths=500,
            df=5.0,
        )
        sim_risk_result = compute_sim_risk(sim, weights, portfolio_value=total_value)
        breaches_sim = risk_budget_check(sim_risk_result)
        breach_list_sim = {k: v for k, v in breaches_sim.items() if v.get("breach")}
        await _store_risk(as_of, sim_risk_result, "simulation", breach_list_sim)

    # Top risk contributors
    sorted_risk = sorted(
        risk_param.component_risk.items(), key=lambda x: abs(x[1]), reverse=True
    )

    logger.info(
        f"Risk: vol={risk_param.total_vol_ann:.1%}, VaR95=${risk_param.var_95_1d:,.0f}, "
        f"CVaR95=${risk_param.cvar_95_1d:,.0f}, E[MDD]={risk_param.expected_max_drawdown_1y:.1%}, "
        f"breaches={len(breach_list)}"
    )

    return {
        "status": "ok",
        "as_of": as_of.isoformat(),
        "portfolio_value": total_value,
        "parametric": {
            "vol": f"{risk_param.total_vol_ann:.1%}",
            "var_95": f"${risk_param.var_95_1d:,.0f}",
            "cvar_95": f"${risk_param.cvar_95_1d:,.0f}",
            "expected_max_drawdown": f"{risk_param.expected_max_drawdown_1y:.1%}",
            "effective_n": f"{risk_param.effective_n:.1f}",
        },
        "simulation": {
            "vol": f"{sim_risk_result.total_vol_ann:.1%}",
            "var_95": f"${sim_risk_result.var_95_1d:,.0f}",
            "cvar_95": f"${sim_risk_result.cvar_95_1d:,.0f}",
        } if sim_risk_result else None,
        "top_risk_contributors": [
            {"symbol": sym, "component_risk": f"{cr:.2%}"}
            for sym, cr in sorted_risk[:5]
        ],
        "budget_breaches": breach_list,
        "_risk_param": risk_param,
    }


async def run_scenario_analysis(
    as_of: Optional[date] = None,
    cov_result: Optional[Dict] = None,
    regime_result: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Stage 4: Run all scenario stress tests."""
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    weights, total_value = await _load_portfolio_weights(as_of)
    if not weights or total_value <= 0:
        return {"status": "no_portfolio", "as_of": as_of.isoformat()}

    cov_lw = cov_result.get("_cov_lw") if cov_result else None
    if cov_lw is None:
        symbols, returns, _ = await _load_returns_matrix(CORE_SYMBOLS, 504, as_of)
        if returns:
            cov_lw = ledoit_wolf(returns, symbols)
        else:
            return {"status": "no_data", "as_of": as_of.isoformat()}

    symbols = cov_lw.symbols
    start_prices = await _load_start_prices(symbols, as_of)

    # Run all scenario simulations
    logger.info(f"Running {len(SCENARIO_TEMPLATES)} scenario stress tests")
    scenario_results = {}
    for name, spec in SCENARIO_TEMPLATES.items():
        try:
            result = run_scenario(
                spec=spec,
                base_cov=cov_lw,
                start_prices=start_prices,
                symbols=symbols,
                n_paths=300,
                df=5.0,
            )
            scenario_results[name] = result
        except Exception:
            logger.exception(f"Scenario {name} failed")

    # Stress test portfolio through scenarios
    stress_results = stress_test_portfolio(weights, scenario_results, total_value)

    # Store each scenario
    for sr in stress_results:
        summary = {}
        if sr.scenario_name in scenario_results:
            summary = scenario_results[sr.scenario_name].get("summary", {})
            # Strip non-serializable simulation object
            summary.pop("simulation", None)
        await _store_scenario(as_of, sr.scenario_name, sr, summary)

    logger.info(
        f"Scenarios: worst={stress_results[0].scenario_name} "
        f"(PnL={stress_results[0].portfolio_pnl_pct:+.1%}), "
        f"best={stress_results[-1].scenario_name} "
        f"(PnL={stress_results[-1].portfolio_pnl_pct:+.1%})"
    )

    return {
        "status": "ok",
        "as_of": as_of.isoformat(),
        "n_scenarios": len(stress_results),
        "results": [
            {
                "scenario": sr.scenario_name,
                "pnl_pct": f"{sr.portfolio_pnl_pct:+.1%}",
                "pnl_dollar": f"${sr.portfolio_pnl:+,.0f}",
                "worst_position": sr.worst_position,
            }
            for sr in stress_results
        ],
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

async def run_simulation_pipeline(
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    """Run the complete simulation pipeline.

    Stages:
    1. Covariance estimation
    2. Regime detection
    3. Risk analysis
    4. Scenario stress tests

    Each stage passes data to the next to avoid re-loading from DB.
    """
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    started = datetime.now(tz=UTC)
    results: Dict[str, Any] = {"as_of": as_of.isoformat()}

    # Stage 1: Covariance
    logger.info("=== Simulation Stage 1: Covariance Estimation ===")
    try:
        cov_result = await run_covariance_estimation(as_of=as_of)
        results["covariance"] = {k: v for k, v in cov_result.items() if not k.startswith("_")}
    except Exception:
        logger.exception("Covariance estimation failed")
        cov_result = {}
        results["covariance"] = {"status": "error"}

    # Stage 2: Regime
    logger.info("=== Simulation Stage 2: Regime Detection ===")
    try:
        regime_result = await run_regime_detection(as_of=as_of, cov_result=cov_result)
        results["regime"] = {k: v for k, v in regime_result.items() if not k.startswith("_")}
    except Exception:
        logger.exception("Regime detection failed")
        regime_result = {}
        results["regime"] = {"status": "error"}

    # Stage 3: Risk
    logger.info("=== Simulation Stage 3: Risk Analysis ===")
    try:
        risk_result = await run_risk_analysis(
            as_of=as_of, cov_result=cov_result, regime_result=regime_result,
        )
        results["risk"] = {k: v for k, v in risk_result.items() if not k.startswith("_")}
    except Exception:
        logger.exception("Risk analysis failed")
        results["risk"] = {"status": "error"}

    # Stage 4: Scenarios
    logger.info("=== Simulation Stage 4: Scenario Stress Tests ===")
    try:
        scenario_result = await run_scenario_analysis(
            as_of=as_of, cov_result=cov_result, regime_result=regime_result,
        )
        results["scenarios"] = scenario_result
    except Exception:
        logger.exception("Scenario analysis failed")
        results["scenarios"] = {"status": "error"}

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    results["elapsed_seconds"] = round(elapsed, 1)

    logger.info(f"Simulation pipeline complete in {elapsed:.1f}s")
    return results
