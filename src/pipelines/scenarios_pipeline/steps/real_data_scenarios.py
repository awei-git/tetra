"""
Generate real data scenarios for backtesting with random windows from recent history.
"""

import logging
import random
from datetime import datetime, date, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from src.pipelines.base import PipelineStep, PipelineContext

logger = logging.getLogger(__name__)


class RealDataScenariosStep(PipelineStep):
    """Generate random real data scenarios from recent history."""
    
    def __init__(self):
        super().__init__("Real Data Scenarios", "Generate random windows from real data")
        
    async def execute(self, context: PipelineContext) -> Any:
        """Generate real data scenarios with random start dates and windows."""
        logger.info("Generating real data scenarios...")
        
        # Configuration
        num_scenarios = 30  # Total scenarios to generate
        windows = [14, 30, 90]  # 2 weeks, 1 month, 3 months in days
        
        # Get the date range for the past year
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        
        # For each window length, generate 10 scenarios
        real_scenarios = []
        scenario_id = 1
        
        for window_days in windows:
            window_name = self._get_window_name(window_days)
            
            for i in range(10):  # 10 scenarios per window
                # Random start date in the past year
                # Make sure we have enough room for the window
                max_start = end_date - timedelta(days=window_days + 7)  # 7 days buffer
                min_start = start_date
                
                # Generate random start date
                days_range = (max_start - min_start).days
                random_days = random.randint(0, days_range)
                scenario_start = min_start + timedelta(days=random_days)
                scenario_end = scenario_start + timedelta(days=window_days)
                
                scenario = {
                    'id': f'real_data_{window_name}_{i+1}',
                    'name': f'Real Data {window_name} Scenario {i+1}',
                    'type': 'real_data',
                    'scenario_type': 'historical',
                    'start_date': scenario_start,
                    'end_date': scenario_end,
                    'window_days': window_days,
                    'window_name': window_name,
                    'description': f'Random {window_name} window from {scenario_start} to {scenario_end}',
                    'metadata': {
                        'window_type': window_name,
                        'window_days': window_days,
                        'scenario_number': i + 1,
                        'data_source': 'real_historical',
                        'volatility_multiplier': 1.0,  # Real data, no adjustment
                        'expected_return': None,  # Will be calculated from actual data
                    },
                    'data': {}
                }
                
                real_scenarios.append(scenario)
                scenario_id += 1
                
        logger.info(f"Generated {len(real_scenarios)} real data scenarios")
        
        # Add to context
        existing_scenarios = context.data.get('scenarios', [])
        existing_scenarios.extend(real_scenarios)
        context.data['scenarios'] = existing_scenarios
        
        context.set_metric('real_data_scenarios_generated', len(real_scenarios))
        
        return context
    
    def _get_window_name(self, days: int) -> str:
        """Convert window days to readable name."""
        if days == 14:
            return "2W"
        elif days == 30:
            return "1M"
        elif days == 90:
            return "3M"
        else:
            return f"{days}D"