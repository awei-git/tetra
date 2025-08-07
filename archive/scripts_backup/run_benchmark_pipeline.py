#!/usr/bin/env python3
"""Run the benchmark pipeline manually."""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up minimal environment
os.environ["DATABASE_URL"] = "postgresql://tetra_user:tetra_password@localhost/tetra"

async def run_pipeline():
    """Run the benchmark pipeline."""
    print("Starting benchmark pipeline...")
    
    # Import here to avoid early config loading
    from src.pipelines.benchmark_pipeline import BenchmarkPipeline
    
    # Create pipeline instance
    pipeline = BenchmarkPipeline()
    
    # Run in daily mode
    result = await pipeline.run(
        mode="daily",
        universe="core",  # Use core symbols for faster testing
        parallel=2  # Reduce parallelism for testing
    )
    
    print(f"\nPipeline result: {result}")
    
    if result.get("status") == "success":
        print("\n✅ Benchmark pipeline completed successfully!")
        print(f"Strategies tested: {result.get('strategies_tested')}")
        print(f"Total backtests: {result.get('total_backtests')}")
    else:
        print(f"\n❌ Pipeline failed: {result.get('error')}")


if __name__ == "__main__":
    # Run the pipeline
    asyncio.run(run_pipeline())