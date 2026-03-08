"""Generate daily market report (PDF).

Usage:
  python scripts/run_report.py                    # latest date, with LLM commentary
  python scripts/run_report.py --as-of 2026-03-07
  python scripts/run_report.py --no-llm           # skip LLM narrative (data-only report)
  python scripts/run_report.py --output /tmp/report.pdf
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import settings
from src.report.llm_clients import create_clients


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily market report")
    parser.add_argument("--as-of", type=str, help="ISO date (YYYY-MM-DD)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM commentary")
    parser.add_argument("--output", type=str, help="Output PDF path")
    parser.add_argument(
        "--llm-provider", type=str, default=None,
        help="Preferred LLM provider (openai, deepseek, gemini)",
    )
    return parser.parse_args()


def get_llm_client(provider: str | None = None):
    """Get a single LLM client for report commentary."""
    clients = create_clients(settings)
    if not clients:
        return None
    if provider and provider in clients:
        return clients[provider]
    for name in ["deepseek", "openai", "gemini"]:
        if name in clients:
            logging.info(f"Using LLM provider for report: {name}")
            return clients[name]
    return None


async def main_async(args: argparse.Namespace) -> None:
    from src.report.generator import generate_report

    as_of = None
    if args.as_of:
        as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date()

    llm_client = None
    if not args.no_llm:
        llm_client = get_llm_client(args.llm_provider)
        if not llm_client:
            logging.warning("No LLM client available, generating data-only report")

    result = await generate_report(
        as_of=as_of,
        llm_client=llm_client,
        output_path=args.output,
    )

    logging.info(f"Report: {result}")
    pdf_path = result.get("pdf_path")
    if pdf_path:
        print(f"\nReport generated: {pdf_path}")

    # Email delivery
    if pdf_path and settings.email_enabled:
        from src.report.delivery import send_email
        recipients = [r.strip() for r in settings.email_recipients.split(",") if r.strip()]
        if recipients and settings.smtp_username and settings.smtp_password:
            sent = await send_email(
                pdf_path=pdf_path,
                recipients=recipients,
                smtp_username=settings.smtp_username,
                smtp_password=settings.smtp_password,
            )
            logging.info(f"Email: {'sent' if sent else 'failed'} to {recipients}")
        else:
            logging.warning("Email skipped: missing recipients or SMTP credentials")

    # Push to Mira bridge + briefing artifacts
    from src.mira.push import push_to_mira
    report_date = as_of or datetime.now(tz=timezone.utc).date()
    mira_result = push_to_mira(result, as_of=report_date)
    logging.info(f"Mira push: {mira_result}")


def main() -> None:
    setup_logging()
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
