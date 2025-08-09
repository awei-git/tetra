#!/usr/bin/env python3
"""
Run strategy analysis with the full configuration file.
Provides easy command-line interface for different analysis modes.
"""

import asyncio
import sys
sys.path.append('/Users/angwei/Repos/tetra')

import argparse
from pathlib import Path

async def main():
    """Main entry point for strategy analysis."""
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive strategy analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test (minimal combinations)
  python run_strategy_analysis.py --mode quick_test
  
  # Run core analysis (main strategies on key symbols)
  python run_strategy_analysis.py --mode core_analysis
  
  # Run full analysis (all combinations - takes hours)
  python run_strategy_analysis.py --mode full_analysis
  
  # Use custom config file
  python run_strategy_analysis.py --config my_config.yaml --mode quick_test
        """
    )
    
    parser.add_argument(
        '--config',
        default='config/analysis_config.yaml',
        help='Path to configuration file (default: config/analysis_config.yaml)'
    )
    
    parser.add_argument(
        '--mode',
        default='quick_test',
        choices=['quick_test', 'core_analysis', 'full_analysis'],
        help='Analysis mode to run (default: quick_test)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration and estimated runtime without running analysis'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    print("="*80)
    print("STRATEGY ANALYSIS RUNNER")
    print("="*80)
    print(f"\nConfiguration file: {config_path}")
    print(f"Analysis mode: {args.mode}")
    
    # Import the runner
    from configurable_analysis import ConfigurableAnalysisRunner
    import yaml
    
    # Load config to show summary
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mode_config = config['modes'][args.mode]
    print(f"\nMode description: {mode_config['description']}")
    
    # Show what will be analyzed
    symbols = config['symbols'][mode_config['symbols']]
    strategies = config['strategies'][mode_config['strategies']]
    windows = config['window_sizes'][mode_config['window_sizes']]
    
    print(f"\nAnalysis scope:")
    print(f"  Symbols: {len(symbols)} symbols")
    print(f"  Strategies: {len(strategies)} strategies")
    print(f"  Window sizes: {len(windows)} sizes")
    print(f"  Date range: {config['analysis']['date_range']['start_year']}-{config['analysis']['date_range']['end_year']}")
    
    # Estimate combinations
    date_range_days = (365 * (config['analysis']['date_range']['end_year'] - 
                             config['analysis']['date_range']['start_year'] + 1))
    step_days = config['analysis']['rolling_windows']['step_days']
    
    total_combinations = 0
    for window in windows:
        rolling_windows = max(1, (date_range_days - window['days']) // step_days)
        total_combinations += rolling_windows * len(symbols) * len(strategies)
    
    print(f"\nEstimated combinations: ~{total_combinations:,}")
    print(f"Estimated runtime: ~{total_combinations / 300:.1f} minutes (at ~5 backtests/sec)")
    
    if args.dry_run:
        print("\n✅ Dry run complete. Use without --dry-run to execute analysis.")
        return
    
    # Confirm before running large analyses
    if args.mode == 'full_analysis' and total_combinations > 10000:
        print(f"\n⚠️  This will run {total_combinations:,} backtests and may take several hours.")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Analysis cancelled.")
            return
    
    print("\n" + "-"*80)
    print("Starting analysis...")
    print("-"*80 + "\n")
    
    try:
        # Create and run analyzer
        runner = ConfigurableAnalysisRunner(str(config_path))
        analysis = await runner.run_analysis(args.mode)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        print(f"\nResults saved to: {runner.output_dir}")
        
        # List generated files
        output_files = list(runner.output_dir.glob('*'))
        if output_files:
            print("\nGenerated files:")
            for file in sorted(output_files):
                size_mb = file.stat().st_size / 1024 / 1024
                print(f"  - {file.name} ({size_mb:.1f} MB)")
        
        # Show top strategies
        if 'strategy_performance' in analysis and analysis['strategy_performance'] is not None:
            print("\nTop 3 Strategies by Average Return:")
            perf = analysis['strategy_performance']
            
            # Calculate score
            perf['score'] = (
                perf[('total_return', 'mean')] * 0.5 +
                perf[('sharpe_ratio', 'mean')] * 0.3 +
                (1 - abs(perf[('max_drawdown', 'mean')])) * 0.2
            )
            
            ranked = perf.sort_values('score', ascending=False)
            
            for i, (strategy, row) in enumerate(ranked.head(3).iterrows()):
                avg_return = row[('total_return', 'mean')] * 100
                sharpe = row[('sharpe_ratio', 'mean')]
                print(f"  {i+1}. {strategy}: {avg_return:.2f}% return, {sharpe:.2f} Sharpe")
        
        print("\n✅ Analysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())