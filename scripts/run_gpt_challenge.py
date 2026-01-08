"""Run GPT challenge pass to critique prior recommendations."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.gpt.challenge import run_gpt_challenge


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT challenge pass")
    parser.add_argument("--session", choices=["pre", "post"], help="Force pre/post session")
    args = parser.parse_args()
    asyncio.run(run_gpt_challenge(session=args.session))


if __name__ == "__main__":
    main()
