"""
Stress test scenario generation step.
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List
from datetime import datetime, date, timedelta
from src.pipelines.base import PipelineStep, PipelineContext
from src.pipelines.scenarios_pipeline.event_definitions import (
    EventScenario,
    STRESS_TEST_SCENARIOS,
    get_tariff_scenarios,
    get_future_scenarios
)

logger = logging.getLogger(__name__)


class StressScenarioStep(PipelineStep):
    """Generate stress test scenarios."""
    
    def __init__(self):
        super().__init__("Stress Scenarios", "Generate stress test scenarios")
    
    async def execute(self, context: PipelineContext) -> Any:
        """Generate stress scenarios."""
        logger.info("Generating stress test scenarios...")
        
        # Get market data baseline from context
        market_data = context.data.get('market_data', {})
        
        # Get stress test definitions from context
        definitions = context.data.get('scenario_definitions', {})
        custom_stress_tests = definitions.get('stress_tests', {})
        
        # Combine pre-defined and custom stress scenarios
        all_stress_scenarios = {
            **STRESS_TEST_SCENARIOS,
            **custom_stress_tests
        }
        
        # Include future scenarios (like tariff impacts)
        future_scenarios = get_future_scenarios()
        all_stress_scenarios.update(future_scenarios)
        
        stress_scenarios = []
        
        # Filter scenarios based on data availability (we only have data from 2015)
        DATA_START_DATE = date(2015, 1, 1)
        
        for scenario_id, event in all_stress_scenarios.items():
            # Skip scenarios that require data before 2015
            if event.start_date < DATA_START_DATE:
                logger.debug(f"Skipping stress scenario {event.name}: starts before data availability ({event.start_date} < {DATA_START_DATE})")
                continue
            
            scenario = self._generate_stress_scenario(
                scenario_id=scenario_id,
                event=event,
                market_data=market_data
            )
            
            if scenario:
                stress_scenarios.append(scenario)
                logger.info(f"Generated stress scenario: {event.name}")
        
        # Add to context
        existing_scenarios = context.data.get('scenarios', [])
        existing_scenarios.extend(stress_scenarios)
        context.data['scenarios'] = existing_scenarios
        
        context.set_metric('stress_scenarios_generated', len(stress_scenarios))
        logger.info(f"Generated {len(stress_scenarios)} stress test scenarios")
        
        return context
    
    def _generate_stress_scenario(
        self,
        scenario_id: str,
        event: EventScenario,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Generate a stress test scenario."""
        
        # Create scenario structure
        scenario = {
            'id': scenario_id,
            'name': event.name,
            'type': 'stress',
            'scenario_type': event.scenario_type,
            'start_date': event.start_date,
            'end_date': event.end_date,
            'description': event.description,
            'metadata': {
                'volatility_multiplier': event.volatility_multiplier,
                'expected_return': event.expected_return,
                'affected_symbols': event.affected_symbols,
                'affected_sectors': event.affected_sectors,
                **((event.metadata or {}))
            },
            'data': {}
        }
        
        # Generate stress scenario data
        scenario['data'] = self._create_stress_data(event, market_data)
        
        return scenario
    
    def _create_stress_data(self, event: EventScenario, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create stress test data using REAL historical data."""
        
        stress_data = {}
        
        # For stress scenarios, use REAL market data from the specified period
        # If the period is in the future or doesn't have data, that's fine - just return empty
        if market_data:
            for symbol, df in market_data.items():
                if not df.empty and 'date' in df.columns:
                    # Filter for the event period
                    mask = (
                        (pd.to_datetime(df['date']).dt.date >= event.start_date) &
                        (pd.to_datetime(df['date']).dt.date <= event.end_date)
                    )
                    period_data = df[mask].copy()
                    
                    if not period_data.empty:
                        # Add metadata about the stress scenario
                        period_data['stress_type'] = event.scenario_type
                        period_data['scenario_phase'] = 'stress'
                        
                        # Convert to dict with all REAL OHLCV data
                        stress_data[symbol] = period_data.to_dict('records')
        
        return stress_data