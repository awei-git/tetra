"""Main entry point for the Metrics Pipeline."""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional, List
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.pipelines.metrics_pipeline.pipeline import MetricsPipeline
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


async def main(
    scenario_filter: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    parallel_workers: int = 8,
    force_recalculate: bool = False,
    log_level: str = "INFO"
):
    """
    Run the Metrics Pipeline.
    
    Args:
        scenario_filter: Filter for specific scenario types
        symbols: List of symbols to process
        parallel_workers: Number of parallel workers
        force_recalculate: Force recalculation of existing metrics
        log_level: Logging level
    """
    # Setup logging
    setup_logging(log_level)
    
    logger.info("=" * 60)
    logger.info("METRICS PIPELINE - STAGE 3")
    logger.info("Pre-calculating indicators and metrics for all scenarios")
    logger.info("=" * 60)
    
    # Create pipeline configuration
    config = {
        'parallel_workers': parallel_workers,
        'batch_size': max(10, 100 // parallel_workers),  # Adjust batch size based on workers
        'storage_dir': 'data/metrics'
    }
    
    # Initialize and run pipeline
    pipeline = MetricsPipeline(config)
    
    try:
        results = await pipeline.run(
            scenario_filter=scenario_filter,
            symbols_filter=symbols,
            force_recalculate=force_recalculate
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("METRICS PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Total Scenarios: {results['total_scenarios']}")
        print(f"Completed: {results['completed']}")
        print(f"Skipped: {results['skipped']}")
        print(f"Failed: {results['failed']}")
        print(f"Execution Time: {results['execution_time']}")
        print(f"Storage Location: {results['storage_location']}")
        print("=" * 60)
        
        # Return success if no failures
        return 0 if results['failed'] == 0 else 1
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Metrics Pipeline - Pre-calculate indicators for all scenarios"
    )
    
    parser.add_argument(
        "--scenario",
        type=str,
        help="Filter scenarios by type (e.g., 'crisis', 'bull', 'monte_carlo')"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Specific symbols to process (default: all symbols in scenarios)"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    
    parser.add_argument(
        "--force-recalculate",
        action="store_true",
        help="Force recalculation even if metrics exist"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    exit_code = asyncio.run(main(
        scenario_filter=args.scenario,
        symbols=args.symbols,
        parallel_workers=args.parallel,
        force_recalculate=args.force_recalculate,
        log_level=args.log_level
    ))
    
    sys.exit(exit_code)