#!/usr/bin/env python3
"""Simple test to verify benchmark pipeline basics."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a minimal working strategy
from src.strats.signal_based import SignalBasedStrategy, SignalRule, SignalCondition, ConditionOperator, PositionSide

def create_simple_strategy():
    """Create a simple working strategy for testing."""
    rules = [
        SignalRule(
            name="simple_test",
            entry_conditions=[
                SignalCondition("sma_20", ConditionOperator.GREATER_THAN, 100.0)
            ],
            exit_conditions=[
                SignalCondition("sma_20", ConditionOperator.LESS_THAN, 95.0)
            ],
            position_side=PositionSide.LONG,
            position_size_factor=1.0
        )
    ]
    
    return SignalBasedStrategy(
        name="Simple Test Strategy",
        signal_rules=rules,
        initial_capital=100000,
        position_size=0.1,
        max_positions=1
    )

print("Creating simple strategy...")
strategy = create_simple_strategy()
print(f"✓ Created strategy: {strategy.name}")

print("\nBenchmark pipeline components:")
print("1. Database tables need to be created (PostgreSQL must be running)")
print("2. Run: psql -U postgres -d tetra -f scripts/create_strategy_tables.sql")
print("3. Pipeline scheduled to run at 8:30 PM daily")
print("4. Frontend strategies tab available at http://localhost:5189/strategies")
print("\nNote: The strategies tab will show data after the benchmark pipeline runs.")