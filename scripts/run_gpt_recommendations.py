"""Run GPT recommendations and store them in the database."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.gpt.recommendations import run_gpt_recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT recommendations")
    parser.add_argument("--session", choices=["pre", "post"], help="Force pre/post session")
    args = parser.parse_args()
    asyncio.run(run_gpt_recommendations(session=args.session))


if __name__ == "__main__":
    main()
