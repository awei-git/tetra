"""Run the quantitative simulation engine standalone.

No LLM required — pure quantitative analysis:
1. Covariance estimation (Ledoit-Wolf + EWMA)
2. HMM regime detection (calm / stressed / crisis)
3. Portfolio risk analysis (VaR, CVaR, drawdown, concentration)
4. Scenario stress tests (7 scenarios: rate shock, credit crisis, etc.)

Usage:
  python scripts/run_simulation.py                    # full pipeline, latest date
  python scripts/run_simulation.py --as-of 2026-03-12
  python scripts/run_simulation.py --stage regime     # regime detection only
  python scripts/run_simulation.py --stage risk       # risk analysis only
  python scripts/run_simulation.py --stage scenarios  # stress tests only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantitative simulation engine")
    parser.add_argument("--as-of", type=str, help="ISO date (YYYY-MM-DD)")
    parser.add_argument(
        "--stage",
        choices=["covariance", "regime", "risk", "scenarios", "all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument(
        "--lookback", type=int, default=504,
        help="Lookback window in trading days (default: 504 ≈ 2 years)",
    )
    parser.add_argument(
        "--paths", type=int, default=500,
        help="Number of Monte Carlo paths (default: 500)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> dict:
    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else None

    from src.simulation.pipeline import (
        run_covariance_estimation,
        run_regime_detection,
        run_risk_analysis,
        run_scenario_analysis,
        run_simulation_pipeline,
    )

    if args.stage == "all":
        return await run_simulation_pipeline(as_of=as_of)

    results = {}

    if args.stage == "covariance":
        results["covariance"] = await run_covariance_estimation(as_of=as_of, lookback=args.lookback)
        # Strip internal state
        results["covariance"] = {k: v for k, v in results["covariance"].items() if not k.startswith("_")}

    elif args.stage == "regime":
        cov_result = await run_covariance_estimation(as_of=as_of, lookback=args.lookback)
        regime_result = await run_regime_detection(as_of=as_of, cov_result=cov_result)
        results["regime"] = {k: v for k, v in regime_result.items() if not k.startswith("_")}

    elif args.stage == "risk":
        cov_result = await run_covariance_estimation(as_of=as_of, lookback=args.lookback)
        regime_result = await run_regime_detection(as_of=as_of, cov_result=cov_result)
        risk_result = await run_risk_analysis(as_of=as_of, cov_result=cov_result, regime_result=regime_result)
        results["risk"] = {k: v for k, v in risk_result.items() if not k.startswith("_")}

    elif args.stage == "scenarios":
        cov_result = await run_covariance_estimation(as_of=as_of, lookback=args.lookback)
        regime_result = await run_regime_detection(as_of=as_of, cov_result=cov_result)
        results["scenarios"] = await run_scenario_analysis(
            as_of=as_of, cov_result=cov_result, regime_result=regime_result,
        )

    return results


def main() -> None:
    setup_logging()
    args = parse_args()

    logging.info(f"Simulation engine: stage={args.stage}, as_of={args.as_of or 'latest'}")
    results = asyncio.run(main_async(args))

    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    if "covariance" in results:
        c = results["covariance"]
        print(f"\nCovariance: {c.get('n_obs', '?')} obs, {len(c.get('symbols', []))} symbols")
        print(f"  Method: {c.get('lw_method', '?')}")

    if "regime" in results:
        r = results["regime"]
        print(f"\nRegime: {r.get('current_regime', '?')}")
        for s in r.get("states", []):
            print(f"  {s['label']}: vol={s['vol']}, freq={s['freq']}")
        print(f"  5-day forecast: {r.get('forecast_5d', {})}")

    if "risk" in results:
        rk = results["risk"]
        if rk.get("status") == "ok":
            p = rk.get("parametric", {})
            print(f"\nRisk (portfolio ${rk.get('portfolio_value', 0):,.0f}):")
            print(f"  Vol: {p.get('vol')}, VaR95: {p.get('var_95')}, CVaR95: {p.get('cvar_95')}")
            print(f"  E[MaxDD]: {p.get('expected_max_drawdown')}, Effective N: {p.get('effective_n')}")
            if rk.get("simulation"):
                s = rk["simulation"]
                print(f"  Sim: Vol={s.get('vol')}, VaR95={s.get('var_95')}, CVaR95={s.get('cvar_95')}")
            for c in rk.get("top_risk_contributors", []):
                print(f"  Risk: {c['symbol']} → {c['component_risk']}")
            if rk.get("budget_breaches"):
                print(f"  BREACHES: {list(rk['budget_breaches'].keys())}")

    if "scenarios" in results:
        sc = results["scenarios"]
        if sc.get("status") == "ok":
            print(f"\nScenarios ({sc.get('n_scenarios', 0)}):")
            for s in sc.get("results", []):
                print(f"  {s['scenario']:25s} → {s['pnl_pct']:>7s} ({s['pnl_dollar']})")

    if "elapsed_seconds" in results:
        print(f"\nCompleted in {results['elapsed_seconds']}s")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        # Strip non-serializable internal state
        clean = {}
        for k, v in results.items():
            if isinstance(v, dict):
                clean[k] = {k2: v2 for k2, v2 in v.items() if not k2.startswith("_")}
            else:
                clean[k] = v
        output_path.write_text(json.dumps(clean, indent=2, default=str))
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
