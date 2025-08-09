#!/usr/bin/env python3
"""
Comprehensive strategy analysis as requested:
For any strategy, for any date in historical data, for any window size (2w, 3m, 1y),
run the strategy for stocks and generate full time series of performance metrics.
"""

import asyncio
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from datetime import date, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

from src.simulators.historical import HistoricalSimulator
from src.simulators.portfolio import Portfolio
from src.simulators.base import SimulationConfig
from src.strats.benchmark import get_all_benchmarks
from src.utils.logging import logger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ComprehensiveAnalyzer:
    """Run comprehensive analysis of all strategies across all time periods."""
    
    def __init__(self):
        self.strategies = get_all_benchmarks()
        self.simulator = HistoricalSimulator()
        self.results = []
        
    async def analyze_strategy_on_stock(
        self, 
        strategy_name: str, 
        symbol: str,
        window_size_days: int,
        window_name: str,
        start_year: int = 2020,
        end_year: int = 2025,
        step_days: int = 14  # Roll every 2 weeks
    ) -> List[Dict]:
        """Analyze a single strategy on a stock with rolling windows."""
        
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            logger.error(f"Strategy {strategy_name} not found")
            return []
        
        results = []
        current_date = date(start_year, 1, 1)
        end_date = date(end_year, 12, 31)
        
        while current_date + timedelta(days=window_size_days) <= end_date:
            window_end = current_date + timedelta(days=window_size_days)
            
            try:
                # Create portfolio
                portfolio = Portfolio(initial_cash=100000.0)
                
                # Configure strategy for this symbol
                if hasattr(strategy, 'set_symbols'):
                    strategy.set_symbols([symbol])
                
                # Run backtest
                result = await self.simulator.run_simulation(
                    portfolio=portfolio,
                    start_date=current_date,
                    end_date=window_end,
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
                
                # Win/Loss streaks
                winning_days = (returns > 0).astype(int)
                if len(winning_days) > 0:
                    groups = (winning_days != winning_days.shift()).cumsum()
                    consecutive_wins = winning_days.groupby(groups).sum()
                    max_consecutive_wins = consecutive_wins.max()
                else:
                    max_consecutive_wins = 0
                
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
                    'max_consecutive_wins': max_consecutive_wins,
                    'final_value': result.final_value,
                    'success': True
                })
                
            except Exception as e:
                logger.warning(f"Failed {strategy_name} on {symbol} from {current_date}: {e}")
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
                    'max_consecutive_wins': 0,
                    'final_value': 100000,
                    'success': False,
                    'error': str(e)
                })
            
            # Move to next window
            current_date += timedelta(days=step_days)
        
        return results

    async def run_full_analysis(
        self,
        symbols: List[str] = None,
        strategies: List[str] = None,
        start_year: int = 2023,
        end_year: int = 2025
    ):
        """Run complete analysis across all combinations."""
        
        if symbols is None:
            symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT']
        
        if strategies is None:
            strategies = list(self.strategies.keys())[:10]  # Top 10 strategies
        
        window_configs = [
            (14, '2_weeks'),
            (90, '3_months'),
            (365, '1_year')
        ]
        
        total_combinations = len(strategies) * len(symbols) * len(window_configs)
        logger.info(f"Starting analysis of {total_combinations} combinations")
        
        all_results = []
        completed = 0
        
        for strategy in strategies:
            for symbol in symbols:
                for window_days, window_name in window_configs:
                    logger.info(f"Analyzing {strategy} on {symbol} with {window_name} windows...")
                    
                    results = await self.analyze_strategy_on_stock(
                        strategy, symbol, window_days, window_name,
                        start_year, end_year
                    )
                    
                    all_results.extend(results)
                    completed += 1
                    
                    logger.info(f"Progress: {completed}/{total_combinations} ({completed/total_combinations*100:.1f}%)")
        
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
        report.append("COMPREHENSIVE STRATEGY PERFORMANCE ANALYSIS")
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
                
                report.append(f"\n#{i+1}. {strategy}")
                report.append(f"    Average Return: {avg_return:.2f}% (±{std_return:.2f}%)")
                report.append(f"    Sharpe Ratio: {sharpe:.2f}")
                report.append(f"    Total Trades: {trades:,}")
        
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
        
        # Market scenario recommendations
        report.append("\n" + "="*80)
        report.append("STRATEGY RECOMMENDATIONS")
        report.append("="*80)
        
        report.append("\nFor Different Market Conditions:")
        
        # Find low volatility strategies
        low_vol = rankings.nsmallest(3, ('max_drawdown', 'mean'))
        report.append("\nLow Risk/Conservative:")
        for strategy, _ in low_vol.iterrows():
            report.append(f"  - {strategy}")
        
        # Find high return strategies
        high_return = rankings.nlargest(3, ('total_return', 'mean'))
        report.append("\nHigh Growth/Aggressive:")
        for strategy, _ in high_return.iterrows():
            report.append(f"  - {strategy}")
        
        # Find consistent strategies
        if len(rankings) > 0:
            consistent = rankings[rankings[('total_return', 'std')] < rankings[('total_return', 'std')].median()]
            consistent = consistent.nlargest(3, ('total_return', 'mean'))
            report.append("\nConsistent Performance:")
            for strategy, _ in consistent.iterrows():
                report.append(f"  - {strategy}")
        
        return "\n".join(report)


async def main():
    """Run the comprehensive analysis."""
    
    print("Starting comprehensive strategy analysis...")
    print("This will take several minutes to complete.\n")
    
    analyzer = ComprehensiveAnalyzer()
    
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
    
    # Run analysis
    df = await analyzer.run_full_analysis(
        symbols=symbols,
        strategies=strategies,
        start_year=2023,
        end_year=2025
    )
    
    # Save raw results
    df.to_csv('/tmp/comprehensive_strategy_analysis.csv', index=False)
    print(f"\nRaw results saved to: /tmp/comprehensive_strategy_analysis.csv")
    
    # Analyze results
    analysis = analyzer.analyze_results(df)
    
    # Generate report
    report = analyzer.generate_report(df, analysis)
    print("\n" + report)
    
    # Save report
    with open('/tmp/comprehensive_strategy_report.txt', 'w') as f:
        f.write(report)
    print(f"\nReport saved to: /tmp/comprehensive_strategy_report.txt")
    
    # Additional analysis - Market scenarios
    print("\n" + "="*80)
    print("MARKET SCENARIO ANALYSIS")
    print("="*80)
    print("\nNote: To test strategies under different market scenarios (bull, bear, crash),")
    print("run the scenario simulator separately. This analysis shows historical performance only.")


if __name__ == "__main__":
    asyncio.run(main())