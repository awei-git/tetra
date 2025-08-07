"""Main entry point for the benchmark pipeline."""

import asyncio
import argparse
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.pipelines.benchmark_pipeline import BenchmarkPipeline
from src.utils.logging import logger


async def main():
    """Run the benchmark pipeline."""
    parser = argparse.ArgumentParser(description="Run the benchmark pipeline")
    parser.add_argument(
        "--mode",
        choices=["daily", "backfill"],
        default="daily",
        help="Pipeline mode: daily for EOD run, backfill for historical"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for backfill mode (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", 
        type=str,
        help="End date for backfill mode (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        help="Specific strategies to run (default: all)"
    )
    parser.add_argument(
        "--universe",
        choices=["core", "all", "large_cap", "tech", "crypto"],
        default="core",
        help="Universe of symbols to test strategies on"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel backtests to run"
    )
    
    args = parser.parse_args()
    
    # Parse dates if provided
    kwargs = {
        "mode": args.mode,
        "universe": args.universe,
        "parallel": args.parallel
    }
    
    if args.start_date:
        kwargs["start_date"] = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    if args.end_date:
        kwargs["end_date"] = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if args.strategies:
        kwargs["strategies"] = args.strategies
    
    # Create and run pipeline
    pipeline = BenchmarkPipeline()
    
    try:
        logger.info(f"Starting benchmark pipeline with args: {kwargs}")
        result = await pipeline.run(**kwargs)
        
        if result["status"] == "success":
            logger.info(f"Benchmark pipeline completed successfully")
            logger.info(f"Strategies tested: {result.get('strategies_tested')}")
            logger.info(f"Best strategy: {result.get('best_strategy')}")
            sys.exit(0)
        else:
            logger.error(f"Benchmark pipeline failed: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error in benchmark pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())