"""
Scenarios Pipeline - Stage 2 of the Tetra data processing architecture.
Generates market scenarios for comprehensive strategy testing.
"""

from .pipeline import ScenariosPipeline
from .event_definitions import (
    ALL_SCENARIOS,
    BULL_MARKET_SCENARIOS,
    CRISIS_SCENARIOS,
    FULL_CYCLE_SCENARIOS,
    STRESS_TEST_SCENARIOS,
    SCENARIO_CATEGORIES,
    EventScenario
)

__all__ = [
    'ScenariosPipeline',
    'ALL_SCENARIOS',
    'BULL_MARKET_SCENARIOS', 
    'CRISIS_SCENARIOS',
    'FULL_CYCLE_SCENARIOS',
    'STRESS_TEST_SCENARIOS',
    'SCENARIO_CATEGORIES',
    'EventScenario'
]