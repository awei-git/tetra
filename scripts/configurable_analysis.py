#!/usr/bin/env python3
"""
Configurable Strategy Analysis Runner
Reads configuration from YAML file and runs analysis accordingly.
"""

import asyncio
import sys
sys.path.append('/Users/angwei/Repos/tetra')

import yaml
import json
from datetime import date, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import time
import os

from src.simulators.historical import HistoricalSimulator
from src.simulators.portfolio import Portfolio
from src.strats.benchmark import get_all_benchmarks
from src.utils.logging import logger


class ConfigurableAnalysisRunner:
    """Run strategy analysis based on configuration file."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.strategies = get_all_benchmarks()
        self.simulator = HistoricalSimulator()
        self.results = []
        
        # Create output directory first
        self.output_dir = Path(self.config['analysis']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging after output directory is created
        self._setup_logging()
        
        # Semaphore for concurrency control
        max_concurrent = self.config['analysis']['parallel']['max_concurrent']
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    def _setup_logging(self):
        """Configure logging based on config."""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        
        # Configure logger
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add file handler if specified
        if log_config.get('log_file'):
            log_file = self.output_dir / log_config['log_file']
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logging.getLogger().addHandler(file_handler)
    
    def get_mode_config(self, mode: str) -> Dict[str, List]:
        """Get configuration for a specific mode."""
        mode_config = self.config['modes'].get(mode, {})
        
        return {
            'symbols': self.config['symbols'][mode_config.get('symbols', 'test')],
            'strategies': self.config['strategies'][mode_config.get('strategies', 'test')],
            'window_sizes': self.config['window_sizes'][mode_config.get('window_sizes', 'test')],
            'scenarios': self.config['market_scenarios'][mode_config.get('scenarios', 'test')]
        }
    
    async def analyze_single_combination(
        self,
        strategy_name: str,
        symbol: str,
        start_date: date,
        end_date: date,
        window_name: str
    ) -> Dict:
        """Analyze a single strategy-symbol-window combination."""
        
        async with self.semaphore:
            try:
                strategy = self.strategies.get(strategy_name)
                if not strategy:
                    raise ValueError(f"Strategy {strategy_name} not found")
                
                # Create portfolio
                initial_cash = 100000.0
                portfolio = Portfolio(initial_cash=initial_cash)
                
                # Configure strategy
                if hasattr(strategy, 'set_symbols'):
                    strategy.set_symbols([symbol])
                
                # Run backtest
                result = await self.simulator.run_simulation(
                    portfolio=portfolio,
                    start_date=start_date,
                    end_date=end_date,
                    strategy=strategy
                )
                
                # Calculate all requested metrics
                metrics = self._calculate_metrics(result, initial_cash)
                
                return {
                    'strategy': strategy_name,
                    'symbol': symbol,
                    'window_size': window_name,
                    'start_date': start_date,
                    'end_date': end_date,
                    'success': True,
                    **metrics
                }
                
            except Exception as e:
                logger.error(f"Error in {strategy_name} on {symbol} ({start_date} to {end_date}): {e}")
                return {
                    'strategy': strategy_name,
                    'symbol': symbol,
                    'window_size': window_name,
                    'start_date': start_date,
                    'end_date': end_date,
                    'success': False,
                    'error': str(e),
                    **{metric: 0 for metric in self.config['metrics']['basic']}
                }
    
    def _calculate_metrics(self, result: Any, initial_cash: float) -> Dict[str, float]:
        """Calculate all requested metrics."""
        metrics = {}
        
        # Basic metrics
        basic_metrics = self.config['metrics']['basic']
        metric_mapping = {
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'volatility': result.volatility,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades
        }
        
        for metric in basic_metrics:
            metrics[metric] = metric_mapping.get(metric, 0)
        
        # Advanced metrics if requested
        if 'advanced' in self.config['metrics']:
            # Calculate additional metrics
            equity_curve = result.equity_curve
            returns = equity_curve.pct_change().dropna() if not equity_curve.empty else pd.Series()
            
            if len(returns) > 0:
                # VaR and CVaR
                metrics['var_95'] = np.percentile(returns, 5)
                metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean() if len(returns[returns <= metrics['var_95']]) > 0 else 0
                
                # Sortino ratio
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std() * np.sqrt(252)
                    metrics['sortino_ratio'] = result.annual_return / downside_std if downside_std > 0 else 0
                else:
                    metrics['sortino_ratio'] = float('inf')
                
                # Calmar ratio
                metrics['calmar_ratio'] = result.annual_return / abs(result.max_drawdown) if result.max_drawdown != 0 else 0
                
                # Profit factor
                metrics['profit_factor'] = result.profit_factor if hasattr(result, 'profit_factor') else 0
                
                # Consecutive wins/losses
                winning_days = (returns > 0).astype(int)
                losing_days = (returns < 0).astype(int)
                
                if len(winning_days) > 0:
                    groups = (winning_days != winning_days.shift()).cumsum()
                    consecutive_wins = winning_days.groupby(groups).sum()
                    metrics['max_consecutive_wins'] = consecutive_wins.max()
                else:
                    metrics['max_consecutive_wins'] = 0
                
                if len(losing_days) > 0:
                    groups = (losing_days != losing_days.shift()).cumsum()
                    consecutive_losses = losing_days.groupby(groups).sum()
                    metrics['max_consecutive_losses'] = consecutive_losses.max()
                else:
                    metrics['max_consecutive_losses'] = 0
            else:
                # Default values for advanced metrics
                for metric in self.config['metrics'].get('advanced', []):
                    if metric not in metrics:
                        metrics[metric] = 0
        
        return metrics
    
    async def run_analysis(self, mode: str = 'quick_test'):
        """Run analysis based on selected mode."""
        
        logger.info(f"Starting analysis in '{mode}' mode")
        logger.info(f"Configuration: {self.config['modes'][mode]['description']}")
        
        # Get mode configuration
        mode_config = self.get_mode_config(mode)
        
        # Calculate total combinations
        date_config = self.config['analysis']['date_range']
        start_year = date_config['start_year']
        end_year = date_config['end_year']
        step_days = self.config['analysis']['rolling_windows']['step_days']
        
        # Estimate number of rolling windows per window size
        total_days = (date(end_year, 12, 31) - date(start_year, 1, 1)).days
        
        total_combinations = 0
        for window_config in mode_config['window_sizes']:
            window_days = window_config['days']
            num_windows = (total_days - window_days) // step_days + 1
            total_combinations += num_windows * len(mode_config['symbols']) * len(mode_config['strategies'])
        
        logger.info(f"Total combinations to analyze: {total_combinations}")
        
        # Run analysis
        all_results = []
        completed = 0
        start_time = time.time()
        
        for strategy_name in mode_config['strategies']:
            for symbol in mode_config['symbols']:
                for window_config in mode_config['window_sizes']:
                    window_name = window_config['name']
                    window_days = window_config['days']
                    
                    # Generate rolling windows
                    tasks = []
                    current_date = date(start_year, 1, 1)
                    end_date = date(end_year, 12, 31)
                    
                    while current_date + timedelta(days=window_days) <= end_date:
                        window_end = current_date + timedelta(days=window_days)
                        
                        task = self.analyze_single_combination(
                            strategy_name, symbol, current_date, window_end, window_name
                        )
                        tasks.append(task)
                        
                        current_date += timedelta(days=step_days)
                    
                    # Run all windows for this combination
                    results = await asyncio.gather(*tasks)
                    all_results.extend(results)
                    
                    completed += len(tasks)
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total_combinations - completed) / rate if rate > 0 else 0
                    
                    logger.info(
                        f"Progress: {completed}/{total_combinations} "
                        f"({completed/total_combinations*100:.1f}%) "
                        f"Rate: {rate:.1f}/sec, ETA: {remaining/60:.1f} min"
                    )
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(all_results)
        
        # Save results
        self._save_results(mode)
        
        # Generate analysis
        analysis = self._analyze_results()
        
        # Generate report
        self._generate_report(analysis, mode)
        
        total_time = time.time() - start_time
        logger.info(f"Analysis complete in {total_time/60:.1f} minutes")
        
        return analysis
    
    def _save_results(self, mode: str):
        """Save raw results to files."""
        if self.config['reporting']['generate_csv']:
            csv_path = self.output_dir / f"{mode}_results.csv"
            self.results_df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}")
        
        if self.config['reporting']['generate_excel']:
            excel_path = self.output_dir / f"{mode}_results.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                self.results_df.to_excel(writer, sheet_name='Raw Results', index=False)
                
                # Add summary sheet
                summary_df = self.results_df.groupby('strategy').agg({
                    'total_return': ['mean', 'std', 'min', 'max'],
                    'sharpe_ratio': 'mean',
                    'max_drawdown': 'mean'
                }).round(4)
                summary_df.to_excel(writer, sheet_name='Strategy Summary')
            
            logger.info(f"Excel report saved to {excel_path}")
    
    def _analyze_results(self) -> Dict:
        """Analyze results and generate insights."""
        analysis = {}
        
        # Filter successful results
        successful = self.results_df[self.results_df['success'] == True].copy()
        
        if len(successful) == 0:
            logger.warning("No successful results to analyze")
            return analysis
        
        # Overall statistics
        analysis['total_backtests'] = len(self.results_df)
        analysis['successful_backtests'] = len(successful)
        analysis['success_rate'] = len(successful) / len(self.results_df)
        
        # Strategy performance
        strategy_perf = successful.groupby('strategy').agg({
            'total_return': ['mean', 'std', 'min', 'max', 'count'],
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'total_trades': 'sum'
        }).round(4)
        
        analysis['strategy_performance'] = strategy_perf
        
        # Performance by window size
        window_perf = successful.groupby('window_size').agg({
            'total_return': ['mean', 'std'],
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean'
        }).round(4)
        
        analysis['window_performance'] = window_perf
        
        # Best strategy for each symbol
        best_by_symbol = {}
        for symbol in successful['symbol'].unique():
            symbol_data = successful[successful['symbol'] == symbol]
            if not symbol_data.empty:
                best = symbol_data.groupby('strategy')['total_return'].mean().idxmax()
                best_return = symbol_data.groupby('strategy')['total_return'].mean().max()
                best_by_symbol[symbol] = (best, best_return)
        
        analysis['best_by_symbol'] = best_by_symbol
        
        return analysis
    
    def _generate_report(self, analysis: Dict, mode: str):
        """Generate analysis report."""
        report = []
        report.append("="*80)
        report.append(f"STRATEGY ANALYSIS REPORT - {mode.upper()} MODE")
        report.append("="*80)
        report.append(f"\nGenerated: {pd.Timestamp.now()}")
        report.append(f"Configuration: {self.config['modes'][mode]['description']}")
        report.append(f"\nTotal Backtests: {analysis.get('total_backtests', 0):,}")
        report.append(f"Successful: {analysis.get('successful_backtests', 0):,} ({analysis.get('success_rate', 0):.1%})")
        
        # Strategy rankings
        if 'strategy_performance' in analysis:
            report.append("\n" + "="*80)
            report.append("STRATEGY RANKINGS")
            report.append("="*80)
            
            perf = analysis['strategy_performance']
            perf['score'] = (
                perf[('total_return', 'mean')] * 0.5 +
                perf[('sharpe_ratio', 'mean')] * 0.3 +
                (1 - abs(perf[('max_drawdown', 'mean')])) * 0.2
            )
            
            ranked = perf.sort_values('score', ascending=False)
            
            for i, (strategy, row) in enumerate(ranked.head(10).iterrows()):
                avg_return = row[('total_return', 'mean')] * 100
                sharpe = row[('sharpe_ratio', 'mean')]
                
                report.append(f"\n#{i+1}. {strategy}")
                report.append(f"    Average Return: {avg_return:.2f}%")
                report.append(f"    Sharpe Ratio: {sharpe:.2f}")
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.output_dir / f"{mode}_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {report_path}")
        
        # Also print to console
        print("\n" + report_text)


async def main():
    """Run configurable analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run configurable strategy analysis")
    parser.add_argument(
        '--config',
        default='config/analysis_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        default='quick_test',
        choices=['quick_test', 'core_analysis', 'full_analysis'],
        help='Analysis mode to run'
    )
    
    args = parser.parse_args()
    
    print(f"Loading configuration from {args.config}")
    print(f"Running in {args.mode} mode\n")
    
    # Run analysis
    runner = ConfigurableAnalysisRunner(args.config)
    analysis = await runner.run_analysis(args.mode)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {runner.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())