#!/usr/bin/env python3
"""Test the full benchmark pipeline with strategy selection and visualization."""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up minimal environment
os.environ["DATABASE_URL"] = "postgresql://tetra_user:tetra_password@localhost/tetra"

async def test_pipeline():
    """Test the benchmark pipeline end-to-end."""
    print("Testing Benchmark Pipeline with Strategy Selection\n")
    print("=" * 60)
    
    # Import here to avoid early config loading
    from src.pipelines.benchmark_pipeline import BenchmarkPipeline
    
    # Create pipeline instance
    pipeline = BenchmarkPipeline()
    
    # Test with different universes
    test_configs = [
        {
            "mode": "daily",
            "universe": "core",
            "parallel": 2,
            "description": "Core symbols (SPY, QQQ, top tech, etc.)"
        },
        {
            "mode": "daily", 
            "universe": "tech",
            "parallel": 2,
            "description": "Tech-focused universe"
        }
    ]
    
    for config in test_configs:
        print(f"\n📊 Testing with {config['description']}")
        print("-" * 40)
        
        result = await pipeline.run(
            mode=config["mode"],
            universe=config["universe"],
            parallel=config["parallel"]
        )
        
        if result.get("status") == "success":
            print(f"✅ Success!")
            print(f"  - Strategies tested: {result.get('strategies_tested')}")
            print(f"  - Total backtests: {result.get('total_backtests')}")
            print(f"  - Best strategy: {result.get('best_strategy')}")
            print(f"  - Execution time: {result.get('execution_time')}s")
            
            # Show metrics summary
            if result.get('metrics_summary'):
                print("\n📈 Metrics Summary:")
                for metric, value in result['metrics_summary'].items():
                    print(f"  - {metric}: {value}")
        else:
            print(f"❌ Failed: {result.get('error')}")
    
    # Test API endpoints
    print("\n\n🌐 Testing API Endpoints")
    print("=" * 60)
    
    import httpx
    
    base_url = "http://localhost:8000/api/v1/strategies"
    
    async with httpx.AsyncClient() as client:
        # Test rankings endpoint
        print("\n📊 Latest Rankings:")
        try:
            response = await client.get(f"{base_url}/rankings/latest?limit=5")
            if response.status_code == 200:
                data = response.json()
                if data.get("rankings"):
                    for rank in data["rankings"]:
                        print(f"  {rank['rank']}. {rank['strategy_name']} "
                              f"(Score: {rank['composite_score']}, "
                              f"Return: {rank['metrics']['total_return']}%)")
                else:
                    print("  No rankings available yet")
        except Exception as e:
            print(f"  API not running or error: {e}")
        
        # Test categories endpoint
        print("\n📂 Strategy Categories:")
        try:
            response = await client.get(f"{base_url}/categories")
            if response.status_code == 200:
                data = response.json()
                for cat in data.get("categories", []):
                    print(f"  - {cat['category']}: {cat['strategy_count']} strategies "
                          f"(Avg Return: {cat['avg_return']}%)")
        except Exception as e:
            print(f"  API not running or error: {e}")
    
    print("\n\n✅ Benchmark pipeline test complete!")
    print("\nTo view results in the WebGUI:")
    print("1. Ensure the API is running: uvicorn src.api.main:app --reload")
    print("2. Navigate to the strategies tab in the WebGUI")
    print("3. View rankings, performance charts, and comparisons")


if __name__ == "__main__":
    asyncio.run(test_pipeline())