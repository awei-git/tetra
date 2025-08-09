#!/usr/bin/env python3
"""
Optimized parallel comprehensive strategy analysis using asyncio concurrency.
Shares database connections efficiently and runs much faster.
"""

import asyncio
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from datetime import date, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
import time
from collections import defaultdict

from src.simulators.historical import HistoricalSimulator
from src.simulators.portfolio import Portfolio
from src.simulators.base import SimulationConfig
from src.strats.benchmark import get_all_benchmarks
from src.utils.logging import logger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OptimizedParallelAnalyzer:
    """Run comprehensive analysis using asyncio for maximum efficiency."""
    
    def __init__(self, max_concurrent=10):
        self.strategies = get_all_benchmarks()
        self.simulator = HistoricalSimulator()
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def analyze_single_window(
        self, 
        strategy_name: str, 
        symbol: str,
        start_date: date,
        end_date: date,
        window_name: str
    ) -> Dict:
        """Analyze a single strategy-symbol-window combination."""
        
        async with self.semaphore:  # Limit concurrent executions
            try:
                strategy = self.strategies.get(strategy_name)
                if not strategy:
                    raise ValueError(f"Strategy {strategy_name} not found")
                
                # Create portfolio
                portfolio = Portfolio(initial_cash=100000.0)
                
                # Configure strategy for this symbol
                if hasattr(strategy, 'set_symbols'):
                    strategy.set_symbols([symbol])
                
                # Run backtest
                result = await self.simulator.run_simulation(
                    portfolio=portfolio,
                    start_date=start_date,
                    end_date=end_date,
                    strategy=strategy
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
                
                return {
                    'strategy': strategy_name,
                    'symbol': symbol,
                    'window_size': window_name,
                    'start_date': start_date,
                    'end_date': end_date,
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
                }
                
            except Exception as e:
                return {
                    'strategy': strategy_name,
                    'symbol': symbol,
                    'window_size': window_name,
                    'start_date': start_date,
                    'end_date': end_date,
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
                }

    async def analyze_strategy_windows(
        self,
        strategy_name: str,
        symbol: str,
        window_days: int,
        window_name: str,
        start_year: int,
        end_year: int,
        step_days: int = 14
    ) -> List[Dict]:
        """Analyze all windows for a strategy-symbol pair."""
        
        tasks = []
        current_date = date(start_year, 1, 1)
        end_date = date(end_year, 12, 31)
        
        while current_date + timedelta(days=window_days) <= end_date:
            window_end = current_date + timedelta(days=window_days)
            
            task = self.analyze_single_window(
                strategy_name, symbol, current_date, window_end, window_name
            )
            tasks.append(task)
            
            current_date += timedelta(days=step_days)
        
        # Run all windows concurrently
        results = await asyncio.gather(*tasks)
        return results

    async def run_optimized_analysis(
        self,
        symbols: List[str] = None,
        strategies: List[str] = None,
        start_year: int = 2023,
        end_year: int = 2025,
        step_days: int = 14
    ) -> pd.DataFrame:
        """Run complete analysis using optimized async approach."""
        
        if symbols is None:
            symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT']
        
        if strategies is None:
            strategies = list(self.strategies.keys())[:10]
        
        window_configs = [
            (14, '2_weeks'),
            (90, '3_months'),
            (365, '1_year')
        ]
        
        total_combinations = len(strategies) * len(symbols) * len(window_configs)
        logger.info(f"Starting optimized parallel analysis of {total_combinations} combinations")
        logger.info(f"Max concurrent tasks: {self.max_concurrent}")
        
        all_results = []
        completed = 0
        start_time = time.time()
        
        # Process each combination
        for strategy in strategies:
            for symbol in symbols:
                for window_days, window_name in window_configs:
                    # Get results for this combination
                    results = await self.analyze_strategy_windows(
                        strategy, symbol, window_days, window_name,
                        start_year, end_year, step_days
                    )
                    
                    all_results.extend(results)
                    completed += 1
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total_combinations - completed) / rate if rate > 0 else 0
                    
                    logger.info(
                        f"Progress: {completed}/{total_combinations} "
                        f"({completed/total_combinations*100:.1f}%) "
                        f"ETA: {remaining/60:.1f} minutes"
                    )
        
        total_time = time.time() - start_time
        logger.info(f"Analysis complete in {total_time/60:.1f} minutes")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        return df

    def analyze_results(self, df: pd.DataFrame) -> Dict:
        """Analyze the full results and generate insights."""
        
        analysis = {}
        
        # Overall statistics
        successful = df[df['success'] == True].copy()
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
        
        # Rank by composite score
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
            'max_drawdown': 'mean',
            'win_rate': 'mean'
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
        successful['start_date'] = pd.to_datetime(successful['start_date'])
        successful['year'] = successful['start_date'].dt.year
        
        yearly_perf = successful.groupby('year').agg({
            'total_return': 'mean',
            'sharpe_ratio': 'mean',
            'win_rate': 'mean'
        }).round(4)
        
        analysis['yearly_performance'] = yearly_perf
        
        # Best strategy for each symbol
        best_by_symbol = {}
        for symbol in successful['symbol'].unique():
            symbol_data = successful[successful['symbol'] == symbol]
            if not symbol_data.empty:
                best_strategy = symbol_data.groupby('strategy')['total_return'].mean().idxmax()
                best_return = symbol_data.groupby('strategy')['total_return'].mean().max()
                best_by_symbol[symbol] = (best_strategy, best_return)
        
        analysis['best_strategy_by_symbol'] = best_by_symbol
        
        # Risk analysis
        analysis['risk_metrics'] = {
            'avg_max_drawdown': successful['max_drawdown'].mean(),
            'worst_drawdown': successful['max_drawdown'].min(),
            'avg_var_95': successful['var_95'].mean(),
            'worst_var_95': successful['var_95'].min(),
            'avg_volatility': successful['volatility'].mean()
        }
        
        # Strategy consistency analysis
        consistency_scores = {}
        for strategy in successful['strategy'].unique():
            strat_data = successful[successful['strategy'] == strategy]
            if len(strat_data) > 10:
                # Calculate consistency as % of positive returns
                positive_pct = (strat_data['total_return'] > 0).mean()
                # Calculate stability as inverse of return std dev
                stability = 1 / (strat_data['total_return'].std() + 0.01)
                consistency_scores[strategy] = {
                    'positive_periods': positive_pct,
                    'stability_score': stability,
                    'combined_score': positive_pct * stability
                }
        
        analysis['consistency_analysis'] = consistency_scores
        
        return analysis

    def generate_detailed_report(self, df: pd.DataFrame, analysis: Dict) -> str:
        """Generate comprehensive report with detailed insights."""
        
        report = []
        report.append("="*80)
        report.append("OPTIMIZED COMPREHENSIVE STRATEGY PERFORMANCE ANALYSIS")
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
            for i, (strategy, row) in enumerate(rankings.head(10).iterrows()):
                avg_return = row[('total_return', 'mean')] * 100
                std_return = row[('total_return', 'std')] * 100
                min_return = row[('total_return', 'min')] * 100
                max_return = row[('total_return', 'max')] * 100
                sharpe = row[('sharpe_ratio', 'mean')]
                max_dd = row[('max_drawdown', 'mean')] * 100
                trades = row[('total_trades', 'sum')]
                count = row[('total_return', 'count')]
                
                report.append(f"\n#{i+1}. {strategy}")
                report.append(f"    Average Return: {avg_return:.2f}% (σ={std_return:.2f}%)")
                report.append(f"    Return Range: [{min_return:.2f}%, {max_return:.2f}%]")
                report.append(f"    Sharpe Ratio: {sharpe:.2f}")
                report.append(f"    Avg Max Drawdown: {max_dd:.2f}%")
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
                std_return = row[('total_return', 'std')] * 100
                sharpe = row[('sharpe_ratio', 'mean')]
                max_dd = row[('max_drawdown', 'mean')] * 100
                win_rate = row[('win_rate', 'mean')] * 100
                
                report.append(f"\n{window}:")
                report.append(f"  Average Return: {avg_return:.2f}% (σ={std_return:.2f}%)")
                report.append(f"  Sharpe Ratio: {sharpe:.2f}")
                report.append(f"  Avg Max Drawdown: {max_dd:.2f}%")
                report.append(f"  Win Rate: {win_rate:.1f}%")
        
        # Performance by year
        report.append("\n" + "="*80)
        report.append("PERFORMANCE BY YEAR")
        report.append("="*80)
        
        yearly_perf = analysis.get('yearly_performance', pd.DataFrame())
        if not yearly_perf.empty:
            for year, row in yearly_perf.iterrows():
                report.append(f"\n{year}:")
                report.append(f"  Average Return: {row['total_return']*100:.2f}%")
                report.append(f"  Average Sharpe: {row['sharpe_ratio']:.2f}")
                report.append(f"  Average Win Rate: {row['win_rate']*100:.1f}%")
        
        # Best strategy for each symbol
        report.append("\n" + "="*80)
        report.append("BEST STRATEGY BY SYMBOL")
        report.append("="*80)
        
        best_by_symbol = analysis['best_strategy_by_symbol']
        for symbol, (strategy, return_val) in sorted(best_by_symbol.items()):
            report.append(f"\n{symbol}: {strategy} ({return_val*100:.2f}% avg return)")
        
        # Risk analysis
        report.append("\n" + "="*80)
        report.append("RISK ANALYSIS")
        report.append("="*80)
        
        risk = analysis['risk_metrics']
        report.append(f"\nAverage Max Drawdown: {risk['avg_max_drawdown']*100:.2f}%")
        report.append(f"Worst Drawdown: {risk['worst_drawdown']*100:.2f}%")
        report.append(f"Average VaR (95%): {risk['avg_var_95']*100:.2f}%")
        report.append(f"Worst VaR (95%): {risk['worst_var_95']*100:.2f}%")
        report.append(f"Average Volatility: {risk['avg_volatility']*100:.2f}%")
        
        # Consistency analysis
        report.append("\n" + "="*80)
        report.append("STRATEGY CONSISTENCY ANALYSIS")
        report.append("="*80)
        
        consistency = analysis.get('consistency_analysis', {})
        if consistency:
            sorted_consistency = sorted(
                consistency.items(), 
                key=lambda x: x[1]['combined_score'], 
                reverse=True
            )
            
            report.append("\nMost Consistent Strategies:")
            for strategy, scores in sorted_consistency[:5]:
                report.append(f"\n{strategy}:")
                report.append(f"  Positive Periods: {scores['positive_periods']*100:.1f}%")
                report.append(f"  Stability Score: {scores['stability_score']:.2f}")
        
        # Key insights
        report.append("\n" + "="*80)
        report.append("KEY INSIGHTS")
        report.append("="*80)
        
        # Find patterns
        successful = df[df['success'] == True]
        if len(successful) > 0:
            # Best overall performer
            best_overall = rankings.index[0] if not rankings.empty else "N/A"
            
            # Most consistent
            if consistency:
                most_consistent = sorted_consistency[0][0]
            else:
                most_consistent = "N/A"
            
            # Best for short vs long term
            short_term = successful[successful['window_size'] == '2_weeks']
            long_term = successful[successful['window_size'] == '1_year']
            
            if not short_term.empty:
                best_short = short_term.groupby('strategy')['total_return'].mean().idxmax()
            else:
                best_short = "N/A"
                
            if not long_term.empty:
                best_long = long_term.groupby('strategy')['total_return'].mean().idxmax()
            else:
                best_long = "N/A"
            
            report.append(f"\n• Best Overall Strategy: {best_overall}")
            report.append(f"• Most Consistent Strategy: {most_consistent}")
            report.append(f"• Best Short-Term (2 weeks): {best_short}")
            report.append(f"• Best Long-Term (1 year): {best_long}")
            
            # Market insights
            report.append("\n• Market Observations:")
            report.append(f"  - Shorter windows show {window_perf.loc['2_weeks', ('total_return', 'std')]/window_perf.loc['1_year', ('total_return', 'std')]:.1f}x more volatility")
            report.append(f"  - Win rates improve from {window_perf.loc['2_weeks', ('win_rate', 'mean')]*100:.1f}% (2w) to {window_perf.loc['1_year', ('win_rate', 'mean')]*100:.1f}% (1y)")
        
        return "\n".join(report)


async def main():
    """Run the optimized parallel comprehensive analysis."""
    
    print("Starting OPTIMIZED parallel comprehensive strategy analysis...")
    print("Using asyncio concurrency for efficient database connection sharing")
    print("This should complete much faster while avoiding connection limits.\n")
    
    analyzer = OptimizedParallelAnalyzer(max_concurrent=10)
    
    # Define what to analyze - start with a smaller set
    symbols = ['SPY', 'QQQ', 'IWM']
    strategies = [
        'buy_and_hold', 
        'golden_cross', 
        'momentum_factor',
        'rsi_reversion',
        'macd_crossover'
    ]
    
    # Run optimized analysis
    start_time = time.time()
    df = await analyzer.run_optimized_analysis(
        symbols=symbols,
        strategies=strategies,
        start_year=2024,  # Shorter period for demo
        end_year=2025,
        step_days=30  # Monthly steps
    )
    
    # Save raw results
    df.to_csv('/tmp/optimized_strategy_analysis.csv', index=False)
    print(f"\nRaw results saved to: /tmp/optimized_strategy_analysis.csv")
    print(f"Total records: {len(df):,}")
    print(f"Analysis time: {(time.time() - start_time)/60:.1f} minutes")
    
    # Analyze results
    analysis = analyzer.analyze_results(df)
    
    # Generate detailed report
    report = analyzer.generate_detailed_report(df, analysis)
    print("\n" + report)
    
    # Save report
    with open('/tmp/optimized_strategy_report.txt', 'w') as f:
        f.write(report)
    print(f"\nDetailed report saved to: /tmp/optimized_strategy_report.txt")
    
    # Summary statistics
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Total backtests completed: {len(df):,}")
    print(f"Success rate: {analysis['success_rate']:.1%}")
    print(f"Time per backtest: {(time.time() - start_time)/len(df):.2f} seconds")
    print(f"Strategies analyzed: {len(strategies)}")
    print(f"Symbols analyzed: {len(symbols)}")
    print(f"Time windows: 2 weeks, 3 months, 1 year")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR FULL ANALYSIS")
    print("="*80)
    print("\n1. This demo analyzed 5 strategies × 3 symbols × 3 windows × ~12 time periods = ~540 backtests")
    print("2. Full analysis would cover all 20 strategies × 18 symbols × 3 windows × ~60 periods = ~65k backtests")
    print("3. With optimization, this would take approximately 2-3 hours")
    print("4. Results would provide complete performance metrics for ranking and selection")
    print("5. Market scenario testing can be run separately on top strategies")


if __name__ == "__main__":
    asyncio.run(main())