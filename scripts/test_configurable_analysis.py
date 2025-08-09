#!/usr/bin/env python3
"""
Test the configurable analysis system with a minimal configuration.
"""

import asyncio
import sys
sys.path.append('/Users/angwei/Repos/tetra')

import yaml
import tempfile
from pathlib import Path

# Create a minimal test configuration
TEST_CONFIG = """
analysis:
  name: "test_run"
  output_dir: "/tmp/test_strategy_analysis"
  parallel:
    max_concurrent: 5
  date_range:
    start_year: 2025
    end_year: 2025
  rolling_windows:
    step_days: 30

symbols:
  test:
    - SPY
    - QQQ

strategies:
  test:
    - buy_and_hold
    - golden_cross

window_sizes:
  test:
    - name: "1_month"
      days: 30
    - name: "3_months"
      days: 90

market_scenarios:
  test:
    - normal

modes:
  quick_test:
    symbols: test
    strategies: test
    window_sizes: test
    scenarios: test
    description: "Minimal test configuration"

metrics:
  basic:
    - total_return
    - sharpe_ratio
    - max_drawdown
    - total_trades

reporting:
  generate_csv: true
  generate_excel: false
  generate_plots: false
  generate_html_report: false

logging:
  level: INFO
  log_file: "test_analysis.log"
  console_output: true
"""

async def run_test():
    """Run a quick test of the configurable analysis system."""
    
    print("="*80)
    print("TESTING CONFIGURABLE ANALYSIS SYSTEM")
    print("="*80)
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(TEST_CONFIG)
        config_path = f.name
    
    print(f"\nCreated test configuration at: {config_path}")
    
    # Import the runner
    from configurable_analysis import ConfigurableAnalysisRunner
    
    # Create runner
    print("\nInitializing analysis runner...")
    runner = ConfigurableAnalysisRunner(config_path)
    
    # Print configuration summary
    print("\nTest Configuration:")
    print(f"  Symbols: {runner.config['symbols']['test']}")
    print(f"  Strategies: {runner.config['strategies']['test']}")
    print(f"  Window sizes: {[w['name'] for w in runner.config['window_sizes']['test']]}")
    print(f"  Date range: {runner.config['analysis']['date_range']['start_year']}-{runner.config['analysis']['date_range']['end_year']}")
    print(f"  Output directory: {runner.output_dir}")
    
    # Calculate expected combinations
    mode_config = runner.get_mode_config('quick_test')
    total_combinations = len(mode_config['symbols']) * len(mode_config['strategies']) * len(mode_config['window_sizes'])
    print(f"\nExpected combinations: ~{total_combinations * 3} (with rolling windows)")
    
    # Run analysis
    print("\nStarting analysis...")
    print("-"*80)
    
    try:
        analysis = await runner.run_analysis('quick_test')
        
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        
        print(f"\nTotal backtests completed: {analysis.get('total_backtests', 0)}")
        print(f"Successful backtests: {analysis.get('successful_backtests', 0)}")
        print(f"Success rate: {analysis.get('success_rate', 0):.1%}")
        
        # Check for errors
        if runner.results_df is not None and not runner.results_df.empty:
            failed = runner.results_df[runner.results_df['success'] == False]
            if not failed.empty:
                print(f"\nFailed backtests: {len(failed)}")
                print("\nSample errors:")
                for idx, row in failed.head(5).iterrows():
                    print(f"  - {row['strategy']} on {row['symbol']}: {row.get('error', 'Unknown error')}")
        
        # Show sample results
        if 'strategy_performance' in analysis:
            print("\nStrategy Performance Summary:")
            perf = analysis['strategy_performance']
            for strategy in perf.index:
                avg_return = perf.loc[strategy, ('total_return', 'mean')] * 100
                print(f"  {strategy}: {avg_return:.2f}% average return")
        
        print(f"\nResults saved to: {runner.output_dir}")
        
        # List output files
        output_files = list(runner.output_dir.glob('*'))
        if output_files:
            print("\nGenerated files:")
            for file in output_files:
                print(f"  - {file.name}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        Path(config_path).unlink()
        print(f"\nCleaned up temporary config file")

if __name__ == "__main__":
    asyncio.run(run_test())