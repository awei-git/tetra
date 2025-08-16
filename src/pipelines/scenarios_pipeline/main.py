"""
Main entry point for Scenarios Pipeline.
"""

import argparse
import asyncio
import logging
from datetime import date, datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pipelines.scenarios_pipeline import ScenariosPipeline
from src.pipelines.scenarios_pipeline.event_definitions import SCENARIO_CATEGORIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_pipeline(args):
    """Run the scenarios pipeline with given arguments."""
    
    # Build configuration from arguments
    config = {
        'scenario_types': [],
        'data': {
            'lookback_years': args.lookback_years,
            'symbols_universe': args.symbols.split(',') if args.symbols else 'core'
        }
    }
    
    # Determine which scenario types to generate
    if args.type == 'all':
        # Disabled stochastic for now as requested
        config['scenario_types'] = ['historical', 'regime', 'stress']
    elif args.type == 'historical':
        config['scenario_types'] = ['historical']
        config['historical'] = {
            'include_bull_markets': not args.no_bull,
            'include_crises': not args.no_crisis,
            'include_full_cycles': args.include_full_cycles
        }
    elif args.type == 'regime':
        config['scenario_types'] = ['regime']
    elif args.type == 'stochastic':
        config['scenario_types'] = ['stochastic']
        config['stochastic'] = {
            'monte_carlo': {
                'enabled': not args.no_monte_carlo,
                'num_scenarios': args.num_scenarios
            },
            'bootstrap': {
                'enabled': not args.no_bootstrap,
                'num_scenarios': args.num_scenarios // 2
            }
        }
    elif args.type == 'stress':
        config['scenario_types'] = ['stress']
        config['stress'] = {
            'severity_levels': [args.severity] if args.severity else ['moderate', 'severe', 'extreme']
        }
    
    # Storage options
    config['storage'] = {
        'save_to_database': not args.dry_run,
        'save_timeseries': args.save_timeseries,
        'save_metadata': True
    }
    
    # Create and run pipeline
    pipeline = ScenariosPipeline(config)
    
    logger.info(f"Running Scenarios Pipeline in '{args.type}' mode")
    if args.dry_run:
        logger.info("DRY RUN MODE - No database writes")
    
    # Parse dates if provided
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    
    # Run pipeline
    results = await pipeline.run(
        start_date=start_date,
        end_date=end_date
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SCENARIOS PIPELINE RESULTS")
    print("="*60)
    print(f"Status: {'SUCCESS' if results['success'] else 'FAILED'}")
    print(f"Scenarios Generated: {results['scenarios_generated']}")
    print(f"Execution Time: {results['execution_time']:.1f} seconds")
    
    if results['scenarios_generated'] > 0:
        print("\nScenario Types Generated:")
        for scenario_type in results['scenario_types']:
            print(f"  - {scenario_type}")
        
        # Show sample scenarios
        print("\nSample Scenarios (first 5):")
        for i, scenario in enumerate(results['scenarios'][:5]):
            print(f"  {i+1}. {scenario.get('name', 'Unknown')}")
            print(f"     Type: {scenario.get('type')}")
            print(f"     Period: {scenario.get('start_date')} to {scenario.get('end_date')}")
            print(f"     Volatility: {scenario.get('volatility_multiplier', 1.0):.1f}x")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run Scenarios Pipeline for market scenario generation'
    )
    
    parser.add_argument(
        '--type',
        type=str,
        default='all',
        choices=['all', 'historical', 'regime', 'stochastic', 'stress'],
        help='Type of scenarios to generate'
    )
    
    parser.add_argument(
        '--lookback-years',
        type=int,
        default=15,
        help='Years of historical data to use (default: 15)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for historical data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for historical data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols (default: core universe)'
    )
    
    # Historical scenario options
    parser.add_argument(
        '--no-bull',
        action='store_true',
        help='Exclude bull market scenarios'
    )
    
    parser.add_argument(
        '--no-crisis',
        action='store_true',
        help='Exclude crisis scenarios'
    )
    
    parser.add_argument(
        '--include-full-cycles',
        action='store_true',
        help='Include full cycle scenarios (crash + recovery)'
    )
    
    # Stochastic scenario options
    parser.add_argument(
        '--num-scenarios',
        type=int,
        default=1000,
        help='Number of stochastic scenarios to generate'
    )
    
    parser.add_argument(
        '--no-monte-carlo',
        action='store_true',
        help='Disable Monte Carlo simulations'
    )
    
    parser.add_argument(
        '--no-bootstrap',
        action='store_true',
        help='Disable bootstrap simulations'
    )
    
    # Stress scenario options
    parser.add_argument(
        '--severity',
        type=str,
        choices=['moderate', 'severe', 'extreme'],
        help='Severity level for stress scenarios'
    )
    
    # Storage options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without saving to database'
    )
    
    parser.add_argument(
        '--save-timeseries',
        action='store_true',
        help='Save full timeseries data (large)'
    )
    
    # Execution options
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers'
    )
    
    parser.add_argument(
        '--list-scenarios',
        action='store_true',
        help='List available scenario definitions and exit'
    )
    
    args = parser.parse_args()
    
    # Handle list scenarios option
    if args.list_scenarios:
        print("\n" + "="*60)
        print("AVAILABLE SCENARIO DEFINITIONS")
        print("="*60)
        
        for category, scenario_keys in SCENARIO_CATEGORIES.items():
            print(f"\n{category.upper()} ({len(scenario_keys)} scenarios):")
            for key in scenario_keys[:5]:  # Show first 5
                print(f"  - {key}")
            if len(scenario_keys) > 5:
                print(f"  ... and {len(scenario_keys) - 5} more")
        
        sys.exit(0)
    
    # Run async pipeline
    try:
        results = asyncio.run(run_pipeline(args))
        sys.exit(0 if results['success'] else 1)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()