"""
Scenario storage step - saves scenarios to parquet files with daily overwrite.
"""

import logging
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
from src.pipelines.base import PipelineStep, PipelineContext

logger = logging.getLogger(__name__)


class ScenarioStorageStep(PipelineStep):
    """Store scenarios to parquet files with daily overwrite."""
    
    def __init__(self):
        super().__init__("Scenario Storage", "Store scenarios to parquet files")
    
    async def execute(self, context: PipelineContext) -> Any:
        """Store scenarios to files."""
        logger.info("Storing scenarios to files...")
        
        scenarios = context.data.get('scenarios', [])
        if not scenarios:
            logger.warning("No scenarios to store")
            return context
        
        # Create storage directory
        storage_dir = Path('data/scenarios')
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Use fixed filenames (overwritten daily)
        metadata_file = storage_dir / 'scenario_metadata.json'
        timeseries_file = storage_dir / 'scenario_timeseries.parquet'
        
        try:
            # Separate metadata and timeseries data
            metadata_list = []
            timeseries_list = []
            
            for scenario in scenarios:
                # Extract metadata
                metadata = {
                    'id': scenario.get('id'),
                    'name': scenario.get('name'),
                    'type': scenario.get('type'),
                    'scenario_type': scenario.get('scenario_type'),
                    'start_date': str(scenario.get('start_date')),
                    'end_date': str(scenario.get('end_date')),
                    'description': scenario.get('description'),
                    'volatility_multiplier': scenario.get('metadata', {}).get('volatility_multiplier', 1.0),
                    'expected_return': scenario.get('metadata', {}).get('expected_return', 0.0),
                    'generated_at': datetime.now().isoformat()
                }
                
                # Add additional metadata fields
                if 'metadata' in scenario:
                    for key in ['affected_symbols', 'affected_sectors', 'catalyst', 
                               'scenario_type', 'phases', 'severity']:
                        if key in scenario['metadata']:
                            metadata[f'meta_{key}'] = scenario['metadata'][key]
                
                metadata_list.append(metadata)
                
                # Extract timeseries data if present
                if 'data' in scenario and scenario['data']:
                    for symbol, timeseries in scenario['data'].items():
                        if isinstance(timeseries, list):
                            for point in timeseries:
                                ts_record = {
                                    'scenario_id': scenario.get('id'),
                                    'symbol': symbol,
                                    **point  # Include all timeseries fields
                                }
                                timeseries_list.append(ts_record)
            
            # Save metadata as JSON (human-readable)
            with open(metadata_file, 'w') as f:
                json.dump({
                    'generated_at': datetime.now().isoformat(),
                    'total_scenarios': len(metadata_list),
                    'scenarios': metadata_list
                }, f, indent=2, default=str)
            
            logger.info(f"Saved scenario metadata to {metadata_file}")
            
            # Save timeseries as parquet (efficient for large data)
            if timeseries_list:
                df_timeseries = pd.DataFrame(timeseries_list)
                df_timeseries.to_parquet(timeseries_file, index=False, compression='snappy')
                logger.info(f"Saved {len(timeseries_list)} timeseries records to {timeseries_file}")
            else:
                logger.info("No timeseries data to save")
            
            # Create a summary file for quick reference
            summary_file = storage_dir / 'scenario_summary.txt'
            with open(summary_file, 'w') as f:
                f.write(f"Scenarios Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Total Scenarios: {len(scenarios)}\n\n")
                
                # Group by type
                by_type = {}
                for scenario in scenarios:
                    stype = scenario.get('scenario_type', scenario.get('type', 'unknown'))
                    if stype not in by_type:
                        by_type[stype] = []
                    by_type[stype].append(scenario.get('name', 'Unnamed'))
                
                for stype, names in sorted(by_type.items()):
                    f.write(f"\n{stype.upper()} ({len(names)} scenarios):\n")
                    f.write(f"{'-'*40}\n")
                    for name in sorted(names)[:10]:  # Show first 10
                        f.write(f"  • {name}\n")
                    if len(names) > 10:
                        f.write(f"  ... and {len(names)-10} more\n")
            
            logger.info(f"Saved scenario summary to {summary_file}")
            
            # Update context with storage info
            storage_status = {
                'scenarios_stored': len(scenarios),
                'metadata_file': str(metadata_file),
                'timeseries_file': str(timeseries_file),
                'summary_file': str(summary_file),
                'storage_errors': []
            }
            
        except Exception as e:
            logger.error(f"Error storing scenarios: {e}")
            storage_status = {
                'scenarios_stored': 0,
                'storage_errors': [str(e)]
            }
        
        context.data['storage_status'] = storage_status
        context.set_metric('scenarios_stored', storage_status['scenarios_stored'])
        
        return context


class ScenarioLoaderStep(PipelineStep):
    """Load previously generated scenarios from storage."""
    
    def __init__(self):
        super().__init__("Scenario Loader", "Load scenarios from storage")
    
    async def execute(self, context: PipelineContext) -> Any:
        """Load scenarios from files."""
        logger.info("Loading scenarios from storage...")
        
        storage_dir = Path('data/scenarios')
        metadata_file = storage_dir / 'scenario_metadata.json'
        timeseries_file = storage_dir / 'scenario_timeseries.parquet'
        
        if not metadata_file.exists():
            logger.warning("No stored scenarios found")
            context.data['scenarios'] = []
            return context
        
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata_data = json.load(f)
            
            scenarios_metadata = metadata_data.get('scenarios', [])
            logger.info(f"Loaded {len(scenarios_metadata)} scenario definitions")
            
            # Load timeseries if exists
            timeseries_by_scenario = {}
            if timeseries_file.exists():
                df_timeseries = pd.read_parquet(timeseries_file)
                
                # Group by scenario_id
                for scenario_id, group in df_timeseries.groupby('scenario_id'):
                    timeseries_by_scenario[scenario_id] = {}
                    
                    # Further group by symbol
                    for symbol, symbol_group in group.groupby('symbol'):
                        # Convert to list of dicts, excluding scenario_id and symbol
                        records = symbol_group.drop(columns=['scenario_id', 'symbol']).to_dict('records')
                        timeseries_by_scenario[scenario_id][symbol] = records
                
                logger.info(f"Loaded timeseries data for {len(timeseries_by_scenario)} scenarios")
            
            # Reconstruct scenarios
            scenarios = []
            for meta in scenarios_metadata:
                scenario = {
                    'id': meta['id'],
                    'name': meta['name'],
                    'type': meta['type'],
                    'scenario_type': meta['scenario_type'],
                    'start_date': meta['start_date'],
                    'end_date': meta['end_date'],
                    'description': meta['description'],
                    'metadata': {
                        'volatility_multiplier': meta.get('volatility_multiplier', 1.0),
                        'expected_return': meta.get('expected_return', 0.0),
                        'loaded_from_cache': True,
                        'cached_at': meta.get('generated_at')
                    }
                }
                
                # Add additional metadata
                for key, value in meta.items():
                    if key.startswith('meta_'):
                        scenario['metadata'][key[5:]] = value
                
                # Add timeseries data if available
                if meta['id'] in timeseries_by_scenario:
                    scenario['data'] = timeseries_by_scenario[meta['id']]
                else:
                    scenario['data'] = {}
                
                scenarios.append(scenario)
            
            context.data['scenarios'] = scenarios
            context.set_metric('scenarios_loaded', len(scenarios))
            
            logger.info(f"Successfully loaded {len(scenarios)} scenarios from storage")
            
            # Show age of cached data
            generated_at = metadata_data.get('generated_at')
            if generated_at:
                cache_age = datetime.now() - datetime.fromisoformat(generated_at)
                logger.info(f"Scenarios were generated {cache_age.total_seconds()/3600:.1f} hours ago")
            
        except Exception as e:
            logger.error(f"Error loading scenarios: {e}")
            context.data['scenarios'] = []
            context.set_metric('scenarios_loaded', 0)
        
        return context