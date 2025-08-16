"""Step 1: Gather all necessary data for assessment."""

import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import pyarrow.parquet as pq
import json

from src.pipelines.base import PipelineStep, PipelineContext
from src.definitions.market_universe import MarketUniverse
from src.definitions.strategies import DEFAULT_STRATEGIES

logger = logging.getLogger(__name__)


class DataGatheringStep(PipelineStep):
    """Gather scenarios, metrics, symbols, and strategies for assessment."""
    
    def __init__(self):
        super().__init__("DataGathering")
    
    async def execute(self, context: PipelineContext) -> None:
        """Gather all data needed for assessment."""
        logger.info("Gathering data for assessment pipeline")
        
        # 1. Load ALL scenarios from the scenarios pipeline output
        scenarios = await self._load_scenarios()
        context.data['scenarios'] = scenarios
        logger.info(f"Loaded {len(scenarios)} scenarios")
        
        # 2. Load ALL symbols from market universe
        symbols = await self._load_symbols()
        context.data['symbols'] = symbols
        logger.info(f"Loaded {len(symbols)} symbols for testing")
        
        # 3. Load ALL strategy configurations
        strategies = await self._load_strategy_configs()
        context.data['strategy_configs'] = strategies
        logger.info(f"Loaded {len(strategies)} strategy configurations")
        
        # 4. Load metrics data for all scenarios
        metrics_data = await self._load_metrics(scenarios)
        context.data['metrics_data'] = metrics_data
        logger.info(f"Loaded metrics for {len(metrics_data)} scenarios")
        
        # Calculate total combinations
        total_combinations = len(strategies) * len(symbols) * len(scenarios)
        logger.info(f"Total combinations to test: {total_combinations:,}")
        context.data['total_combinations'] = total_combinations
    
    async def _load_scenarios(self) -> List[Dict[str, Any]]:
        """Load ALL scenario definitions from scenarios pipeline output."""
        scenarios_dir = Path('data/scenarios')
        metadata_file = scenarios_dir / 'scenario_metadata.json'
        
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Scenario metadata not found at {metadata_file}. "
                "Please run the scenarios pipeline first."
            )
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        scenarios = []
        
        # Handle both old and new metadata formats
        if 'scenarios' in metadata:
            # New format with scenarios array
            for scenario_info in metadata['scenarios']:
                scenarios.append({
                    'name': scenario_info.get('name', scenario_info.get('id', 'unknown')),
                    'type': scenario_info.get('type', 'historical'),
                    'start_date': scenario_info.get('start_date'),
                    'end_date': scenario_info.get('end_date'),
                    'description': scenario_info.get('description', ''),
                    'metadata': scenario_info
                })
        else:
            # Old format with direct items
            for scenario_name, scenario_info in metadata.items():
                if isinstance(scenario_info, dict):
                    scenarios.append({
                        'name': scenario_name,
                        'type': scenario_info.get('type', 'historical'),
                        'start_date': scenario_info.get('start_date'),
                        'end_date': scenario_info.get('end_date'),
                        'description': scenario_info.get('description', ''),
                        'metadata': scenario_info
                    })
        
        return scenarios
    
    async def _load_symbols(self) -> List[str]:
        """Load ALL symbols from market universe."""
        universe = MarketUniverse()
        
        # Get ALL symbols from the universe
        all_symbols = universe.get_all_symbols()
        
        # Remove duplicates while preserving order (if any)
        seen = set()
        unique_symbols = []
        for symbol in all_symbols:
            if symbol not in seen:
                seen.add(symbol)
                unique_symbols.append(symbol)
        
        return unique_symbols
    
    async def _load_strategy_configs(self) -> List[Dict[str, Any]]:
        """Load ALL strategy configurations from DEFAULT_STRATEGIES."""
        strategies = []
        
        for name, config in DEFAULT_STRATEGIES.items():
            strategies.append({
                'name': name,
                'config': config,
                'category': config.category.value if hasattr(config.category, 'value') else str(config.category),
                'description': config.description,
                'parameters': config.parameters
            })
        
        return strategies
    
    async def _load_metrics(self, scenarios: List[Dict]) -> Dict[str, pd.DataFrame]:
        """Load pre-calculated metrics for ALL scenarios."""
        metrics_dir = Path('data/metrics')
        
        if not metrics_dir.exists():
            raise FileNotFoundError(
                f"Metrics directory not found at {metrics_dir}. "
                "Please run the metrics pipeline first."
            )
        
        metrics_data = {}
        missing_metrics = []
        
        for scenario in scenarios:
            scenario_name = scenario['name']
            metrics_file = metrics_dir / f"{scenario_name}_metrics.parquet"
            
            if metrics_file.exists():
                try:
                    df = pq.read_table(metrics_file).to_pandas()
                    metrics_data[scenario_name] = df
                    logger.debug(f"Loaded metrics for scenario: {scenario_name}")
                except Exception as e:
                    logger.error(f"Failed to load metrics for {scenario_name}: {e}")
                    missing_metrics.append(scenario_name)
            else:
                logger.error(f"No metrics file found for scenario: {scenario_name}")
                missing_metrics.append(scenario_name)
        
        if missing_metrics:
            logger.warning(
                f"Missing metrics for {len(missing_metrics)} scenarios: {missing_metrics}. "
                f"These scenarios will be skipped."
            )
            # Remove missing scenarios from the list
            scenarios = [s for s in scenarios if s['name'] not in missing_metrics]
        
        return metrics_data