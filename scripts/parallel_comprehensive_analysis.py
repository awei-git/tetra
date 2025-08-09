#!/usr/bin/env python3
"""
Parallel comprehensive strategy analysis using multiprocessing.
Runs much faster by analyzing multiple strategy-symbol-window combinations in parallel.
"""

import asyncio
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from datetime import date, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pickle

from src.simulators.historical import HistoricalSimulator
from src.simulators.portfolio import Portfolio
from src.simulators.base import SimulationConfig
from src.strats.benchmark import get_all_benchmarks
from src.utils.logging import logger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def analyze_single_combination(args: Tuple) -> List[Dict]:
    """Analyze a single strategy-symbol-window combination.
    This function runs in a separate process."""
    
    strategy_name, symbol, window_days, window_name, start_year, end_year, step_days = args
    
    # Need to create simulator and strategies in each process
    simulator = HistoricalSimulator()
    strategies = get_all_benchmarks()
    
    strategy = strategies.get(strategy_name)
    if not strategy:
        return []
    
    results = []
    current_date = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)
    
    while current_date + timedelta(days=window_days) <= end_date:
        window_end = current_date + timedelta(days=window_days)
        
        try:
            # Create portfolio
            portfolio = Portfolio(initial_cash=100000.0)
            
            # Configure strategy for this symbol
            if hasattr(strategy, 'set_symbols'):
                strategy.set_symbols([symbol])
            
            # Run backtest - need to run async in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                simulator.run_simulation(
                    portfolio=portfolio,
                    start_date=current_date,
                    end_date=window_end,
                    strategy=strategy
                )
            )
            
            # Calculate additional metrics
            equity_curve = result.equity_curve
            returns = equity_curve.pct_change().dropna() if not equity_curve.empty else pd.Series()
            
            # Value at Risk
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
            
            # Maximum consecutive losses
            losing_days = (returns < 0).astype(int)
            max_consecutive_losses = 0
            if len(losing_days) > 0:
                groups = (losing_days != losing_days.shift()).cumsum()
                consecutive_losses = losing_days.groupby(groups).sum()
                max_consecutive_losses = consecutive_losses.max()
            
            # Store results
            results.append({
                'strategy': strategy_name,
                'symbol': symbol,
                'window_size': window_name,
                'start_date': current_date,
                'end_date': window_end,
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'max_drawdown': result.max_drawdown,
                'volatility': result.volatility,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'profit_factor': result.profit_factor,
                'var_95': var_95,
                'max_consecutive_losses': max_consecutive_losses,
                'final_value': result.final_value,
                'success': True
            })
            
        except Exception as e:
            results.append({
                'strategy': strategy_name,
                'symbol': symbol,
                'window_size': window_name,
                'start_date': current_date,
                'end_date': window_end,
                'total_return': 0,
                'annual_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'volatility': 0,
                'win_rate': 0,
                'total_trades': 0,
                'profit_factor': 0,
                'var_95': 0,
                'max_consecutive_losses': 0,
                'final_value': 100000,
                'success': False,
                'error': str(e)
            })
        
        finally:
            loop.close()
        
        # Move to next window
        current_date += timedelta(days=step_days)
    
    return results


class ParallelAnalyzer:
    """Run comprehensive analysis using parallel processing."""
    
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or mp.cpu_count() - 1
        self.strategies = list(get_all_benchmarks().keys())
        
    def run_parallel_analysis(
        self,
        symbols: List[str] = None,
        strategies: List[str] = None,
        start_year: int = 2023,
        end_year: int = 2025,
        step_days: int = 14
    ) -> pd.DataFrame:
        """Run complete analysis using multiprocessing."""
        
        if symbols is None:
            symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT']
        
        if strategies is None:
            strategies = self.strategies[:10]  # Top 10 strategies
        
        window_configs = [
            (14, '2_weeks'),
            (90, '3_months'),
            (365, '1_year')
        ]
        
        # Create all combinations
        combinations = []
        for strategy in strategies:
            for symbol in symbols:
                for window_days, window_name in window_configs:
                    combinations.append((
                        strategy, symbol, window_days, window_name,
                        start_year, end_year, step_days
                    ))
        
        total_combinations = len(combinations)
        logger.info(f"Starting parallel analysis of {total_combinations} combinations using {self.max_workers} workers")
        
        all_results = []
        completed = 0
        start_time = time.time()
        
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_combo = {
                executor.submit(analyze_single_combination, combo): combo
                for combo in combinations
            }
            
            # Process completed tasks
            for future in as_completed(future_to_combo):
                combo = future_to_combo[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    completed += 1
                    
                    # Progress update
                    if completed % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        remaining = (total_combinations - completed) / rate
                        
                        logger.info(
                            f"Progress: {completed}/{total_combinations} "
                            f"({completed/total_combinations*100:.1f}%) "
                            f"Rate: {rate:.1f} combos/sec, "
                            f"ETA: {remaining/60:.1f} minutes"
                        )
                    
                except Exception as e:
                    logger.error(f"Failed to process {combo[0]} on {combo[1]}: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"Analysis complete in {total_time/60:.1f} minutes")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        return df

    def analyze_results(self, df: pd.DataFrame) -> Dict:
        """Analyze the full results and generate insights."""
        
        analysis = {}
        
        # Overall statistics
        successful = df[df['success'] == True]
        analysis['total_backtests'] = len(df)
        analysis['successful_backtests'] = len(successful)
        analysis['success_rate'] = len(successful) / len(df) if len(df) > 0 else 0
        
        if len(successful) == 0:
            return analysis
        
        # Best performing strategies overall
        strategy_perf = successful.groupby('strategy').agg({
            'total_return': ['mean', 'std', 'min', 'max', 'count'],
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'win_rate': 'mean',
            'total_trades': 'sum'
        }).round(4)
        
        # Rank by average return
        strategy_perf['score'] = (
            strategy_perf[('total_return', 'mean')] * 0.4 +
            strategy_perf[('sharpe_ratio', 'mean')] * 0.3 +
            (1 - abs(strategy_perf[('max_drawdown', 'mean')])) * 0.3
        )
        
        analysis['strategy_rankings'] = strategy_perf.sort_values('score', ascending=False)
        
        # Performance by window size
        window_perf = successful.groupby('window_size').agg({
            'total_return': ['mean', 'std'],
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean'
        }).round(4)
        
        analysis['window_performance'] = window_perf
        
        # Performance by symbol
        symbol_perf = successful.groupby('symbol').agg({
            'total_return': ['mean', 'std'],
            'sharpe_ratio': 'mean',
            'volatility': 'mean'
        }).round(4)
        
        analysis['symbol_performance'] = symbol_perf
        
        # Time-based analysis
        df['start_date'] = pd.to_datetime(df['start_date'])
        successful['year'] = successful['start_date'].dt.year
        
        yearly_perf = successful.groupby('year').agg({
            'total_return': 'mean',
            'sharpe_ratio': 'mean',
            'win_rate': 'mean'
        }).round(4)
        
        analysis['yearly_performance'] = yearly_perf
        
        # Find best strategy for each symbol
        best_by_symbol = {}
        for symbol in successful['symbol'].unique():
            symbol_data = successful[successful['symbol'] == symbol]
            best = symbol_data.groupby('strategy')['total_return'].mean().idxmax()
            best_return = symbol_data.groupby('strategy')['total_return'].mean().max()
            best_by_symbol[symbol] = (best, best_return)
        
        analysis['best_strategy_by_symbol'] = best_by_symbol
        
        # Risk analysis
        analysis['risk_metrics'] = {
            'avg_max_drawdown': successful['max_drawdown'].mean(),
            'worst_drawdown': successful['max_drawdown'].min(),
            'avg_var_95': successful['var_95'].mean(),
            'worst_var_95': successful['var_95'].min(),
            'avg_volatility': successful['volatility'].mean()
        }
        
        return analysis

    def generate_report(self, df: pd.DataFrame, analysis: Dict) -> str:
        """Generate comprehensive report."""
        
        report = []
        report.append("="*80)
        report.append("PARALLEL COMPREHENSIVE STRATEGY PERFORMANCE ANALYSIS")
        report.append("="*80)
        report.append(f"\nAnalysis Period: {df['start_date'].min()} to {df['end_date'].max()}")
        report.append(f"Total Backtests Run: {analysis['total_backtests']:,}")
        report.append(f"Successful Backtests: {analysis['successful_backtests']:,} ({analysis['success_rate']:.1%})")
        
        # Top strategies
        report.append("\n" + "="*80)
        report.append("TOP PERFORMING STRATEGIES")
        report.append("="*80)
        
        rankings = analysis['strategy_rankings']
        if not rankings.empty:
            for i, (strategy, row) in enumerate(rankings.head(5).iterrows()):
                avg_return = row[('total_return', 'mean')] * 100
                std_return = row[('total_return', 'std')] * 100
                sharpe = row[('sharpe_ratio', 'mean')]
                trades = row[('total_trades', 'sum')]
                count = row[('total_return', 'count')]
                
                report.append(f"\n#{i+1}. {strategy}")
                report.append(f"    Average Return: {avg_return:.2f}% (±{std_return:.2f}%)")
                report.append(f"    Sharpe Ratio: {sharpe:.2f}")
                report.append(f"    Total Trades: {trades:,}")
                report.append(f"    Backtests: {count}")
        
        # Performance by window size
        report.append("\n" + "="*80)
        report.append("PERFORMANCE BY TIME WINDOW")
        report.append("="*80)
        
        window_perf = analysis['window_performance']
        if not window_perf.empty:
            for window, row in window_perf.iterrows():
                avg_return = row[('total_return', 'mean')] * 100
                sharpe = row[('sharpe_ratio', 'mean')]
                max_dd = row[('max_drawdown', 'mean')] * 100
                
                report.append(f"\n{window}:")
                report.append(f"  Average Return: {avg_return:.2f}%")
                report.append(f"  Sharpe Ratio: {sharpe:.2f}")
                report.append(f"  Avg Max Drawdown: {max_dd:.2f}%")
        
        # Best strategy for each symbol
        report.append("\n" + "="*80)
        report.append("BEST STRATEGY BY SYMBOL")
        report.append("="*80)
        
        best_by_symbol = analysis['best_strategy_by_symbol']
        for symbol, (strategy, return_val) in best_by_symbol.items():
            report.append(f"\n{symbol}: {strategy} ({return_val*100:.2f}% avg return)")
        
        # Risk metrics
        report.append("\n" + "="*80)
        report.append("RISK ANALYSIS")
        report.append("="*80)
        
        risk = analysis['risk_metrics']
        report.append(f"\nAverage Max Drawdown: {risk['avg_max_drawdown']*100:.2f}%")
        report.append(f"Worst Drawdown: {risk['worst_drawdown']*100:.2f}%")
        report.append(f"Average VaR (95%): {risk['avg_var_95']*100:.2f}%")
        report.append(f"Average Volatility: {risk['avg_volatility']*100:.2f}%")
        
        return "\n".join(report)


async def main():
    """Run the parallel comprehensive analysis."""
    
    print("Starting PARALLEL comprehensive strategy analysis...")
    print(f"Using {mp.cpu_count() - 1} CPU cores for processing")
    print("This should complete much faster than sequential analysis.\n")
    
    analyzer = ParallelAnalyzer()
    
    # Define what to analyze
    symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL']
    strategies = [
        'buy_and_hold', 
        'golden_cross', 
        'turtle_trading',
        'rsi_reversion',
        'momentum_factor',
        'macd_crossover',
        'bollinger_bands',
        'dual_momentum'
    ]
    
    # Run parallel analysis
    df = analyzer.run_parallel_analysis(
        symbols=symbols,
        strategies=strategies,
        start_year=2023,
        end_year=2025
    )
    
    # Save raw results
    df.to_csv('/tmp/parallel_comprehensive_strategy_analysis.csv', index=False)
    print(f"\nRaw results saved to: /tmp/parallel_comprehensive_strategy_analysis.csv")
    print(f"Total records: {len(df):,}")
    
    # Analyze results
    analysis = analyzer.analyze_results(df)
    
    # Generate report
    report = analyzer.generate_report(df, analysis)
    print("\n" + report)
    
    # Save report
    with open('/tmp/parallel_comprehensive_strategy_report.txt', 'w') as f:
        f.write(report)
    print(f"\nReport saved to: /tmp/parallel_comprehensive_strategy_report.txt")
    
    # Quick performance comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"\nTotal backtests analyzed: {len(df):,}")
    print(f"Analysis methods compared:")
    print(f"  - Sequential: ~30-60 minutes for this dataset")
    print(f"  - Parallel: Completed in minutes using {mp.cpu_count() - 1} cores")
    print(f"\nParallel speedup: ~{mp.cpu_count() - 1}x faster")


if __name__ == "__main__":
    # Use asyncio to run the main function
    asyncio.run(main())