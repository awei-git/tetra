"""
Main entry point for data pipeline - designed for Airflow integration
"""

import asyncio
import argparse
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.pipelines.data_pipeline import DataPipeline
from src.pipelines.base import PipelineStatus
from src.utils.logging import logger


async def run_pipeline(args):
    """Run the data pipeline with given arguments"""
    
    # Configure pipeline based on mode
    if args.mode == "daily":
        # Daily update: fetch today's data
        target_date = args.date or date.today()
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
            
        config = {
            "mode": "daily",
            "start_date": target_date,
            "end_date": target_date,
            "symbols": args.symbols,  # None means use default symbols
            "news_provider": args.news_provider or "newsapi",
            "parallel": not args.sequential,
            "skip_steps": args.skip_steps,
        }
        logger.info(f"Running daily update for {target_date}")
        
    else:  # backfill mode
        # Parse dates
        if args.days:
            # Backfill by number of days
            end_date = args.end_date or date.today()
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
            start_date = end_date - timedelta(days=args.days)
        else:
            # Backfill by date range
            start_date = args.start_date
            end_date = args.end_date or date.today()
            
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                
        config = {
            "mode": "backfill",
            "start_date": start_date,
            "end_date": end_date,
            "symbols": args.symbols,
            "news_provider": args.news_provider or "alphavantage",
            "parallel": not args.sequential,
            "skip_steps": args.skip_steps,
        }
        logger.info(f"Running backfill from {start_date} to {end_date}")
    
    # Create and run pipeline
    pipeline = DataPipeline()
    context = await pipeline.run(**config)
    
    # Return appropriate exit code
    if context.status == PipelineStatus.SUCCESS:
        return 0
    elif context.status == PipelineStatus.PARTIAL:
        return 1
    else:
        return 2


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Data Pipeline - Ingest market data, economic indicators, events, and news",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Daily update (for Airflow/cron at 8pm)
  python -m src.pipelines.data_pipeline.main --mode=daily
  
  # Daily update for specific date
  python -m src.pipelines.data_pipeline.main --mode=daily --date=2024-01-15
  
  # Backfill last 30 days
  python -m src.pipelines.data_pipeline.main --mode=backfill --days=30
  
  # Backfill specific date range
  python -m src.pipelines.data_pipeline.main --mode=backfill --start-date=2024-01-01 --end-date=2024-01-31
  
  # Backfill specific symbols
  python -m src.pipelines.data_pipeline.main --mode=backfill --days=7 --symbols AAPL MSFT GOOGL
  
  # Skip specific steps
  python -m src.pipelines.data_pipeline.main --mode=daily --skip-steps news_sentiment event_data
        """
    )
    
    # Mode selection (required)
    parser.add_argument(
        "--mode", 
        choices=["daily", "backfill"], 
        required=True,
        help="Pipeline mode: daily update or historical backfill"
    )
    
    # Date parameters
    parser.add_argument(
        "--date",
        type=str,
        help="Target date for daily mode (YYYY-MM-DD). Default: today"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for backfill mode (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", 
        type=str,
        help="End date for backfill mode (YYYY-MM-DD). Default: today"
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Number of days to backfill (alternative to start-date)"
    )
    
    # Data selection
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Specific symbols to process. Default: configured symbol list"
    )
    parser.add_argument(
        "--skip-steps",
        nargs="+",
        choices=["market_data", "economic_data", "event_data", "news_sentiment", "data_quality"],
        help="Steps to skip"
    )
    
    # Provider options
    parser.add_argument(
        "--news-provider",
        choices=["newsapi", "alphavantage"],
        help="News data provider. Default: newsapi for daily, alphavantage for backfill"
    )
    
    # Execution options
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run steps sequentially instead of in parallel"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "backfill":
        if not args.days and not args.start_date:
            parser.error("Backfill mode requires either --days or --start-date")
    
    # Run pipeline
    try:
        if args.dry_run:
            print(f"Would run {args.mode} pipeline with:")
            print(f"  Date range: {args.start_date or 'calculated'} to {args.end_date or 'today'}")
            print(f"  Symbols: {args.symbols or 'default list'}")
            print(f"  News provider: {args.news_provider or 'default'}")
            print(f"  Skip steps: {args.skip_steps or 'none'}")
            return 0
        else:
            exit_code = asyncio.run(run_pipeline(args))
            return exit_code
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())