"""
Historical scenario extraction step.
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List
from datetime import datetime, timedelta
from src.pipelines.base import PipelineStep, PipelineContext
from src.pipelines.scenarios_pipeline.event_definitions import (
    EventScenario, 
    BULL_MARKET_SCENARIOS,
    CRISIS_SCENARIOS,
    FULL_CYCLE_SCENARIOS,
    get_scenarios_by_type
)

logger = logging.getLogger(__name__)


class HistoricalScenarioStep(PipelineStep):
    """Extract historical market scenarios."""
    
    def __init__(self):
        super().__init__("Historical Scenarios", "Extract historical scenarios")
    
    async def execute(self, context: PipelineContext) -> Any:
        """Extract historical scenarios."""
        logger.info("Extracting historical scenarios...")
        
        # Get market data from context
        market_data = context.data.get('market_data', {})
        
        # Get scenario definitions from context
        definitions = context.data.get('scenario_definitions', {})
        historical_events = definitions.get('historical_events', {})
        
        # Combine with pre-defined scenarios
        all_historical = {
            **BULL_MARKET_SCENARIOS,
            **CRISIS_SCENARIOS,
            **FULL_CYCLE_SCENARIOS,
            **historical_events
        }
        
        historical_scenarios = []
        
        # Filter scenarios based on data availability (we only have data from 2015)
        from datetime import date
        DATA_START_DATE = date(2015, 1, 1)
        
        for scenario_id, event in all_historical.items():
            # Skip future scenarios
            if event.start_date >= datetime.now().date():
                continue
            
            # Skip scenarios that require data before 2015
            if event.start_date < DATA_START_DATE:
                logger.debug(f"Skipping scenario {event.name}: starts before data availability ({event.start_date} < {DATA_START_DATE})")
                continue
                
            scenario = self._extract_scenario_data(
                scenario_id=scenario_id,
                event=event,
                market_data=market_data
            )
            
            if scenario:
                historical_scenarios.append(scenario)
                logger.info(f"Extracted scenario: {event.name} ({event.start_date} to {event.end_date})")
        
        # Add to context
        existing_scenarios = context.data.get('scenarios', [])
        existing_scenarios.extend(historical_scenarios)
        context.data['scenarios'] = existing_scenarios
        
        context.set_metric('historical_scenarios_extracted', len(historical_scenarios))
        logger.info(f"Extracted {len(historical_scenarios)} historical scenarios")
        
        return context
    
    def _extract_scenario_data(
        self, 
        scenario_id: str,
        event: EventScenario,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Extract data for a specific historical scenario."""
        
        # Create scenario structure
        scenario = {
            'id': scenario_id,
            'name': event.name,
            'type': 'historical',
            'scenario_type': event.scenario_type,
            'start_date': event.start_date,
            'end_date': event.end_date,
            'description': event.description,
            'metadata': {
                'volatility_multiplier': event.volatility_multiplier,
                'expected_return': event.expected_return,
                'key_dates': event.key_dates,
                'affected_symbols': event.affected_symbols,
                'affected_sectors': event.affected_sectors,
                **((event.metadata or {}))
            },
            'data': {}
        }
        
        # If we have actual market data, extract the period
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
                        # Apply volatility adjustments if needed
                        if event.volatility_multiplier != 1.0 and 'returns' in period_data.columns:
                            period_data['adjusted_volatility'] = (
                                period_data['returns'].std() * event.volatility_multiplier
                            )
                        
                        scenario['data'][symbol] = period_data.to_dict('records')
        
        # If no market data, just return empty data
        # Scenarios without data will be filtered out or handled downstream
        
        return scenario
