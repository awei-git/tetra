"""
Main Scenarios Pipeline implementation.
"""

from typing import Dict, List, Any, Optional
from datetime import date, datetime, timedelta
import logging
from pathlib import Path

from src.pipelines.base import Pipeline, PipelineContext
from .steps import (
    DataLoadingStep,
    HistoricalScenarioStep,
    RegimeDetectionStep,
    StochasticScenarioStep,
    StressScenarioStep,
    ScenarioValidationStep,
    ScenarioStorageStep
)
from .steps.real_data_scenarios import RealDataScenariosStep
from .event_definitions import (
    BULL_MARKET_SCENARIOS,
    CRISIS_SCENARIOS,
    FULL_CYCLE_SCENARIOS,
    STRESS_TEST_SCENARIOS,
    SCENARIO_CATEGORIES
)

logger = logging.getLogger(__name__)


class ScenariosPipeline(Pipeline):
    """
    Pipeline for generating market scenarios from historical data.
    
    Creates different market environments for comprehensive strategy testing:
    - Historical event periods (bull markets, crises, full cycles)
    - Market regime detection (trending, ranging, volatile)
    - Stochastic simulations (Monte Carlo, bootstrap)
    - Stress test scenarios (hypothetical extreme events)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize scenarios pipeline."""
        super().__init__("Scenarios Pipeline")
        
        # Default configuration
        default_config = {
            'scenario_types': ['historical', 'regime', 'stochastic', 'stress'],
            'historical': {
                'include_bull_markets': True,
                'include_crises': True,
                'include_full_cycles': True,
                'context_days_before': 30,
                'context_days_after': 30
            },
            'regime': {
                'min_duration_days': 60,
                'regime_types': ['bull', 'bear', 'high_vol', 'low_vol', 'ranging']
            },
            'stochastic': {
                'monte_carlo': {
                    'enabled': True,
                    'num_scenarios': 1000,
                    'time_horizon_days': 252
                },
                'bootstrap': {
                    'enabled': True,
                    'num_scenarios': 500,
                    'block_size_days': 20
                }
            },
            'stress': {
                'include_standard_tests': True,
                'severity_levels': ['moderate', 'severe', 'extreme']
            },
            'data': {
                'lookback_years': 15,  # Extended for full cycles
                'symbols_universe': 'core',  # 'core', 'all', or list of symbols
                'data_source': 'database'  # 'database' or 'api'
            },
            'validation': {
                'min_data_coverage': 0.95,
                'check_statistical_properties': True,
                'validate_market_constraints': True
            },
            'storage': {
                'save_to_database': True,
                'save_timeseries': False,  # Full timeseries data (large)
                'save_metadata': True
            }
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize steps based on configuration
        self._initialize_steps()
    
    def _initialize_steps(self):
        """Initialize pipeline steps based on configuration."""
        # Always load data first
        self.add_step(DataLoadingStep())
        
        # Add scenario generation steps based on config
        scenario_types = self.config.get('scenario_types', [])
        
        # Always add real data scenarios for recent history
        self.add_step(RealDataScenariosStep())
        
        if 'historical' in scenario_types:
            self.add_step(HistoricalScenarioStep())
        
        if 'regime' in scenario_types:
            self.add_step(RegimeDetectionStep())
        
        if 'stochastic' in scenario_types:
            self.add_step(StochasticScenarioStep())
        
        if 'stress' in scenario_types:
            self.add_step(StressScenarioStep())
        
        # Always validate and store
        self.add_step(ScenarioValidationStep())
        
        if self.config['storage']['save_to_database']:
            self.add_step(ScenarioStorageStep())
    
    async def setup(self) -> PipelineContext:
        """Setup pipeline context."""
        context = PipelineContext()
        context.data['config'] = self.config
        context.data['scenarios'] = []  # Will collect all generated scenarios
        
        # Set date range for data loading
        end_date = date.today()
        lookback_years = self.config['data']['lookback_years']
        start_date = end_date - timedelta(days=lookback_years * 365)
        
        context.data['start_date'] = start_date
        context.data['end_date'] = end_date
        
        # Load scenario definitions
        context.data['scenario_definitions'] = {
            'bull_markets': BULL_MARKET_SCENARIOS,
            'crises': CRISIS_SCENARIOS,
            'full_cycles': FULL_CYCLE_SCENARIOS,
            'stress_tests': STRESS_TEST_SCENARIOS,
            'categories': SCENARIO_CATEGORIES
        }
        
        logger.info(f"Scenarios Pipeline initialized")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Scenario types to generate: {self.config['scenario_types']}")
        
        return context
    
    async def run(self, 
                  scenario_types: Optional[List[str]] = None,
                  start_date: Optional[date] = None,
                  end_date: Optional[date] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Run the scenarios pipeline.
        
        Args:
            scenario_types: Types of scenarios to generate
            start_date: Start date for historical data
            end_date: End date for historical data
            **kwargs: Additional parameters
            
        Returns:
            Pipeline results with generated scenarios
        """
        # Override config if parameters provided
        if scenario_types:
            self.config['scenario_types'] = scenario_types
        
        # Run pipeline through base class
        context = await super().run(**kwargs)
        
        # Compile results
        results = {
            'success': context.status == 'success',
            'scenarios_generated': len(context.data.get('scenarios', [])),
            'scenario_types': context.data.get('scenario_types_generated', []),
            'execution_time': context.metrics.get('duration_seconds'),
            'scenarios': context.data.get('scenarios', []),
            'validation_results': context.data.get('validation_results', {}),
            'storage_status': context.data.get('storage_status', {})
        }
        
        # Log summary
        logger.info(f"Scenarios Pipeline completed")
        logger.info(f"Generated {results['scenarios_generated']} scenarios")
        
        if results['scenarios_generated'] > 0:
            self._log_scenario_summary(results['scenarios'])
        
        return results
    
    def _log_scenario_summary(self, scenarios: List[Dict]):
        """Log summary of generated scenarios."""
        by_type = {}
        for scenario in scenarios:
            scenario_type = scenario.get('type', 'unknown')
            by_type[scenario_type] = by_type.get(scenario_type, 0) + 1
        
        logger.info("Scenarios by type:")
        for scenario_type, count in by_type.items():
            logger.info(f"  {scenario_type}: {count}")
        
        # Log some interesting statistics
        if scenarios:
            volatilities = [s.get('volatility_multiplier', 1.0) for s in scenarios]
            returns = [s.get('expected_return', 0.0) for s in scenarios]
            
            logger.info(f"Volatility range: {min(volatilities):.1f}x to {max(volatilities):.1f}x")
            logger.info(f"Return range: {min(returns):.1%} to {max(returns):.1%}")