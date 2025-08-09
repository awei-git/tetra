#!/usr/bin/env python3
"""Format benchmark pipeline results into a comprehensive report."""

import json
import sys
from datetime import datetime
from typing import Dict, List, Any

def parse_log_results(log_file: str) -> Dict[str, Any]:
    """Parse results from the benchmark log file."""
    results = {
        "strategies_tested": 0,
        "successful": 0,
        "failed": 0,
        "execution_time": 0.0,
        "strategy_results": {},
        "top_strategies": [],
        "category_winners": {},
        "error_summary": {}
    }
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Extract strategy collection info
        if "Collected" in line and "strategies:" in line:
            try:
                strategies = line.split('[')[1].split(']')[0].replace("'", "").split(', ')
                results["all_strategies"] = strategies
                results["strategies_tested"] = len(strategies)
            except:
                pass
        
        # Extract backtest results
        if "Completed backtest for" in line:
            try:
                parts = line.split("Completed backtest for ")[1]
                strategy_name = parts.split(":")[0]
                metrics = parts.split(": ")[1]
                
                # Parse metrics
                return_val = float(metrics.split("Return=")[1].split("%")[0])
                sharpe = float(metrics.split("Sharpe=")[1].split(",")[0])
                trades = int(metrics.split("Trades=")[1].strip())
                
                results["strategy_results"][strategy_name] = {
                    "return": return_val,
                    "sharpe": sharpe,
                    "trades": trades,
                    "status": "success"
                }
            except:
                pass
        
        # Extract errors
        if "Error in backtest for" in line:
            try:
                strategy = line.split("Error in backtest for ")[1].split(":")[0]
                error = line.split(": ", 2)[2].strip()
                results["strategy_results"][strategy] = {
                    "status": "failed",
                    "error": error
                }
                results["error_summary"][strategy] = error
            except:
                pass
        
        # Extract summary stats
        if "Backtest complete:" in line:
            try:
                stats = line.split("Backtest complete: ")[1]
                stats_dict = eval(stats)  # Safe here as it's our own output
                results["successful"] = stats_dict.get("successful", 0)
                results["failed"] = stats_dict.get("failed", 0)
                results["execution_time"] = stats_dict.get("execution_time", 0)
            except:
                pass
    
    return results

def generate_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive benchmark report."""
    report = []
    
    # Header
    report.append("=" * 80)
    report.append("TETRA BENCHMARK PIPELINE REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    
    # Executive Summary
    report.append("## EXECUTIVE SUMMARY")
    report.append("-" * 40)
    report.append(f"Total Strategies Tested: {results['strategies_tested']}")
    report.append(f"Successful Backtests: {results['successful']}")
    report.append(f"Failed Backtests: {results['failed']}")
    if results['strategies_tested'] > 0:
        report.append(f"Success Rate: {results['successful'] / results['strategies_tested'] * 100:.1f}%")
    else:
        report.append(f"Success Rate: N/A")
    report.append(f"Total Execution Time: {results['execution_time']:.2f} seconds")
    report.append("")
    
    # Performance Rankings
    report.append("## PERFORMANCE RANKINGS")
    report.append("-" * 40)
    
    # Sort strategies by return
    successful_strategies = {k: v for k, v in results['strategy_results'].items() 
                           if v.get('status') == 'success'}
    
    if successful_strategies:
        sorted_by_return = sorted(successful_strategies.items(), 
                                key=lambda x: x[1].get('return', 0), 
                                reverse=True)
        
        report.append("### Top 10 by Total Return:")
        for i, (name, metrics) in enumerate(sorted_by_return[:10], 1):
            report.append(f"{i:2d}. {name:30s} Return: {metrics['return']:6.2f}% | "
                         f"Sharpe: {metrics['sharpe']:5.2f} | Trades: {metrics['trades']:3d}")
        report.append("")
        
        # Sort by Sharpe ratio
        sorted_by_sharpe = sorted(successful_strategies.items(), 
                                key=lambda x: x[1].get('sharpe', 0), 
                                reverse=True)
        
        report.append("### Top 10 by Sharpe Ratio:")
        for i, (name, metrics) in enumerate(sorted_by_sharpe[:10], 1):
            report.append(f"{i:2d}. {name:30s} Sharpe: {metrics['sharpe']:5.2f} | "
                         f"Return: {metrics['return']:6.2f}% | Trades: {metrics['trades']:3d}")
        report.append("")
    
    # Category Analysis
    report.append("## STRATEGY CATEGORY ANALYSIS")
    report.append("-" * 40)
    
    # Categorize strategies
    categories = {
        'Passive': ['buy_and_hold'],
        'Trend Following': ['golden_cross', 'turtle_trading', 'donchian_breakout', 'dual_momentum'],
        'Mean Reversion': ['rsi_reversion', 'bollinger_bands'],
        'Momentum': ['momentum_factor', 'macd_crossover', 'volume_momentum', 'mega_cap_momentum', 'ai_growth'],
        'Rotation': ['sector_rotation'],
        'Alternative': ['volatility', 'crypto'],
        'Event Driven': ['dividend', 'earnings'],
        'Intraday': ['morning_breakout'],
        'Multi-Asset': ['global_macro', 'all_weather']
    }
    
    for category, strategy_names in categories.items():
        # Find best performer in category
        category_results = {}
        for strategy, metrics in results['strategy_results'].items():
            base_strategy = strategy.split('_')[0] if '_' in strategy else strategy
            for s in strategy_names:
                if s in base_strategy:
                    category_results[strategy] = metrics
                    break
        
        if category_results:
            best = max(category_results.items(), 
                      key=lambda x: x[1].get('return', -999) if x[1].get('status') == 'success' else -999)
            
            if best[1].get('status') == 'success':
                report.append(f"{category:15s} Winner: {best[0]:25s} "
                            f"(Return: {best[1]['return']:.2f}%, Sharpe: {best[1]['sharpe']:.2f})")
    
    report.append("")
    
    # Error Summary
    if results['error_summary']:
        report.append("## ERROR SUMMARY")
        report.append("-" * 40)
        for strategy, error in results['error_summary'].items():
            report.append(f"- {strategy}: {error}")
        report.append("")
    
    # Detailed Results
    report.append("## DETAILED RESULTS")
    report.append("-" * 40)
    report.append(f"{'Strategy':<35s} {'Status':<10s} {'Return %':>10s} {'Sharpe':>10s} {'Trades':>8s}")
    report.append("-" * 80)
    
    for strategy, metrics in sorted(results['strategy_results'].items()):
        if metrics.get('status') == 'success':
            report.append(f"{strategy:<35s} {'Success':<10s} "
                         f"{metrics['return']:>10.2f} {metrics['sharpe']:>10.2f} "
                         f"{metrics['trades']:>8d}")
        else:
            report.append(f"{strategy:<35s} {'Failed':<10s} "
                         f"{'N/A':>10s} {'N/A':>10s} {'N/A':>8s}")
    
    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    """Main function to generate the report."""
    log_file = "/tmp/benchmark_full_test.log"
    
    # Check if we have a log file from command line
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    try:
        results = parse_log_results(log_file)
        report = generate_report(results)
        print(report)
        
        # Also save to file
        with open("/tmp/benchmark_report.txt", "w") as f:
            f.write(report)
        
        print(f"\nReport also saved to: /tmp/benchmark_report.txt")
        
    except FileNotFoundError:
        print(f"Error: Could not find log file: {log_file}")
        print("Usage: python format_benchmark_results.py [log_file]")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()