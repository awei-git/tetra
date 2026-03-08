"""Run novel analysis pipeline (Phase 1 + Phase 2 + Phase 3).

Phase 1:
1. Narrative Fragmentation Index (keyword + LLM)
2. Polymarket Lead-Lag Scanner (Granger + LLM semantic filter)

Phase 2:
3. Insider Trading Signal (Form 4 pattern detection)
4. Earnings Network / Supply Chain cascade
5. Analyst Network Alpha (co-coverage momentum)

Phase 3:
6. LLM Adversarial Debate (3 LLMs with information asymmetry)
7. Portfolio mark-to-market + recommendation tracker

Unified:
8. Meta-Signal Layer (LLM regime assessment + unified signals)

Usage:
  python scripts/run_analysis.py                  # run all, latest date
  python scripts/run_analysis.py --as-of 2026-03-06
  python scripts/run_analysis.py --stage narrative # run single stage
  python scripts/run_analysis.py --stage debate    # Phase 3 adversarial debate
  python scripts/run_analysis.py --stage portfolio # Portfolio update
  python scripts/run_analysis.py --no-llm         # skip LLM calls (quant only)
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

from config.config import settings
from src.report.llm_clients import create_clients

UTC = timezone.utc


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run novel analysis pipeline")
    parser.add_argument("--as-of", type=str, help="ISO date (YYYY-MM-DD)")
    parser.add_argument(
        "--stage",
        choices=["narrative", "polymarket", "insider", "earnings_net",
                 "analyst_net", "debate", "portfolio", "scenarios", "meta",
                 "phase1", "phase2", "phase3", "all"],
        default="all",
        help="Which stage to run",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM calls (quantitative analysis only)",
    )
    parser.add_argument(
        "--llm-provider", type=str, default=None,
        help="Preferred LLM provider (openai, deepseek, gemini)",
    )
    return parser.parse_args()


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def get_llm_client(provider: str | None = None):
    """Get a single LLM client for analysis."""
    clients = create_clients(settings)
    if not clients:
        return None

    if provider and provider in clients:
        return clients[provider]

    # Preference order: deepseek (cheap), openai (capable), gemini (fallback)
    for name in ["deepseek", "openai", "gemini"]:
        if name in clients:
            logging.info(f"Using LLM provider: {name}")
            return clients[name]
    return None


def get_all_llm_clients() -> dict:
    """Get all available LLM clients (for debate)."""
    return create_clients(settings)


async def run_pipeline(as_of: date | None, stage: str, use_llm: bool, llm_provider: str | None) -> dict:
    """Run the analysis pipeline."""
    results = {}
    started = datetime.now(tz=UTC)

    llm_client = get_llm_client(llm_provider) if use_llm else None
    if use_llm and not llm_client:
        logging.warning("No LLM clients available, running quantitative-only mode")

    phase1_stages = ("narrative", "polymarket", "phase1", "all")
    phase2_stages = ("insider", "earnings_net", "analyst_net", "phase2", "all")

    # Stage 1: Narrative Fragmentation
    if stage in ("narrative",) + phase1_stages[2:]:
        logging.info("=== Stage 1: Narrative Fragmentation Index ===")
        from src.analysis.narrative import run_narrative_analysis
        try:
            results["narrative"] = await run_narrative_analysis(
                as_of=as_of, llm_client=llm_client,
            )
        except Exception:
            logging.exception("Narrative analysis failed")
            results["narrative"] = {"status": "error"}

    # Stage 2: Polymarket Lead-Lag
    if stage in ("polymarket",) + phase1_stages[2:]:
        logging.info("=== Stage 2: Polymarket Lead-Lag Scanner ===")
        from src.analysis.polymarket import run_polymarket_scanner
        try:
            results["polymarket"] = await run_polymarket_scanner(
                as_of=as_of, llm_client=llm_client,
            )
        except Exception:
            logging.exception("Polymarket scanner failed")
            results["polymarket"] = {"status": "error"}

    # Stage 3: Insider Trading Signal
    if stage in ("insider",) + phase2_stages[3:]:
        logging.info("=== Stage 3: Insider Trading Signal ===")
        from src.analysis.insider_signal import run_insider_signal
        try:
            results["insider"] = await run_insider_signal(as_of=as_of)
        except Exception:
            logging.exception("Insider signal failed")
            results["insider"] = {"status": "error"}

    # Stage 4: Earnings Network / Supply Chain
    if stage in ("earnings_net",) + phase2_stages[3:]:
        logging.info("=== Stage 4: Earnings Network ===")
        from src.analysis.earnings_network import run_earnings_network
        try:
            results["earnings_network"] = await run_earnings_network(as_of=as_of)
        except Exception:
            logging.exception("Earnings network failed")
            results["earnings_network"] = {"status": "error"}

    # Stage 5: Analyst Network Alpha
    if stage in ("analyst_net",) + phase2_stages[3:]:
        logging.info("=== Stage 5: Analyst Network Alpha ===")
        from src.analysis.analyst_network import run_analyst_network
        try:
            results["analyst_network"] = await run_analyst_network(as_of=as_of)
        except Exception:
            logging.exception("Analyst network failed")
            results["analyst_network"] = {"status": "error"}

    # Stage 6: Adversarial Debate (uses all LLM clients)
    if stage in ("debate", "phase3", "all"):
        if use_llm:
            logging.info("=== Stage 6: LLM Adversarial Debate ===")
            from src.analysis.debate import run_debate
            try:
                all_clients = get_all_llm_clients()
                debate_result = await run_debate(as_of=as_of, llm_clients=all_clients)
                results["debate"] = debate_result

                # Create tracked recommendations from debate
                if debate_result.get("status") == "success":
                    from src.portfolio.manager import create_recommendations_from_debate
                    synthesis = debate_result.get("synthesis", {})
                    if synthesis:
                        n = await create_recommendations_from_debate(as_of or date.today(), synthesis)
                        logging.info(f"Created {n} tracked recommendations from debate")
            except Exception:
                logging.exception("Adversarial debate failed")
                results["debate"] = {"status": "error"}
        else:
            logging.info("Skipping debate (requires LLM)")

    # Stage 7: Portfolio update
    if stage in ("portfolio", "phase3", "all"):
        logging.info("=== Stage 7: Portfolio Update ===")
        from src.portfolio.manager import run_portfolio_update
        try:
            results["portfolio"] = await run_portfolio_update(as_of=as_of)
        except Exception:
            logging.exception("Portfolio update failed")
            results["portfolio"] = {"status": "error"}

    # Stage 8: Forward Scenario Analysis (requires LLM + portfolio)
    if stage in ("scenarios", "phase3", "all"):
        if use_llm:
            logging.info("=== Stage 8: Forward Scenario Analysis ===")
            from src.analysis.scenarios import run_scenario_analysis
            try:
                results["scenarios"] = await run_scenario_analysis(
                    as_of=as_of, llm_client=llm_client,
                )
            except Exception:
                logging.exception("Scenario analysis failed")
                results["scenarios"] = {"status": "error"}
        else:
            logging.info("Skipping scenarios (requires LLM)")

    # Stage 9: Meta-Signal (requires LLM, runs last — uses all signals)
    if stage in ("meta", "all"):
        logging.info("=== Stage 9: LLM Meta-Signal Layer ===")
        from src.analysis.meta_signal import run_meta_signal
        try:
            results["meta_signal"] = await run_meta_signal(
                as_of=as_of, llm_client=llm_client,
            )
        except Exception:
            logging.exception("Meta-signal analysis failed")
            results["meta_signal"] = {"status": "error"}

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    results["elapsed_seconds"] = round(elapsed, 1)

    # Log LLM stats
    if llm_client:
        stats = llm_client.get_stats()
        results["llm_stats"] = stats
        logging.info(f"LLM stats: {stats}")

    return results


def main() -> None:
    setup_logging()
    args = parse_args()
    as_of = parse_date(args.as_of)

    logging.info(
        f"Starting analysis pipeline: stage={args.stage}, "
        f"as_of={as_of or 'latest'}, llm={'off' if args.no_llm else 'on'}"
    )

    results = asyncio.run(run_pipeline(
        as_of=as_of,
        stage=args.stage,
        use_llm=not args.no_llm,
        llm_provider=args.llm_provider,
    ))

    logging.info(f"Pipeline complete in {results.get('elapsed_seconds', '?')}s")

    # Print summary
    for stage_name in ("narrative", "polymarket", "insider", "earnings_network",
                       "analyst_network", "debate", "portfolio", "scenarios",
                       "meta_signal"):
        if stage_name in results:
            r = results[stage_name]
            status = r.get("status", "unknown")
            logging.info(f"  {stage_name}: {status}")


if __name__ == "__main__":
    main()
