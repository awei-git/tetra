#!/usr/bin/env python3
"""Test script to run benchmark pipeline with minimal dependencies."""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_pipeline():
    """Run a simple test of the benchmark pipeline."""
    print("Testing benchmark pipeline...")
    
    # Check if we can import the pipeline
    try:
        from src.pipelines.benchmark_pipeline import BenchmarkPipeline
        print("✓ Successfully imported BenchmarkPipeline")
    except ImportError as e:
        print(f"✗ Failed to import BenchmarkPipeline: {e}")
        return
    
    # Check if we can import benchmark strategies
    try:
        from src.strats.benchmark import get_core_benchmarks
        strategies = get_core_benchmarks()
        print(f"✓ Found {len(strategies)} core benchmark strategies")
        for name in list(strategies.keys())[:5]:
            print(f"  - {name}")
    except ImportError as e:
        print(f"✗ Failed to import benchmark strategies: {e}")
        return
    
    print("\nBenchmark pipeline components are ready.")
    print("\nTo run the full pipeline:")
    print("1. Ensure PostgreSQL is running")
    print("2. Create the strategy tables using: psql -U postgres -d tetra -f scripts/create_strategy_tables.sql")
    print("3. The pipeline will run automatically at 8:30 PM daily via launchd")
    print("4. Or run manually: python src/pipelines/benchmark_pipeline/main.py")


if __name__ == "__main__":
    asyncio.run(test_pipeline())