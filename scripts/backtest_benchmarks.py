#!/usr/bin/env python3
"""Backtest and compare strategies against benchmarks."""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.strats.benchmark import (
    get_benchmark_strategy,
    get_all_benchmarks,
    get_core_benchmarks,
    get_benchmarks_by_style
)
from src.strats.base import BaseStrategy


def compare_strategies(
    user_strategy: BaseStrategy,
    benchmarks: Dict[str, BaseStrategy],
    market_data: pd.DataFrame,
    signals: pd.DataFrame = None
) -> pd.DataFrame:
    """Compare a user strategy against benchmark strategies.
    
    Args:
        user_strategy: The strategy to evaluate
        benchmarks: Dictionary of benchmark strategies
        market_data: OHLCV data for backtesting
        signals: Pre-computed technical signals
        
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    # Backtest user strategy
    print(f"Backtesting user strategy: {user_strategy.name}")
    user_metrics = backtest_strategy(user_strategy, market_data, signals)
    user_metrics['Strategy'] = user_strategy.name
    user_metrics['Type'] = 'User'
    results.append(user_metrics)
    
    # Backtest each benchmark
    for name, benchmark in benchmarks.items():
        print(f"Backtesting benchmark: {benchmark.name}")
        benchmark_metrics = backtest_strategy(benchmark, market_data, signals)
        benchmark_metrics['Strategy'] = benchmark.name
        benchmark_metrics['Type'] = 'Benchmark'
        results.append(benchmark_metrics)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results)
    
    # Add relative performance
    user_return = user_metrics['Total Return']
    comparison_df['Excess Return'] = comparison_df['Total Return'] - user_return
    comparison_df['Relative Sharpe'] = comparison_df['Sharpe Ratio'] / user_metrics['Sharpe Ratio'] - 1
    
    return comparison_df


def backtest_strategy(
    strategy: BaseStrategy,
    market_data: pd.DataFrame,
    signals: pd.DataFrame = None
) -> Dict[str, Any]:
    """Run a simple backtest for a strategy.
    
    This is a simplified backtest for demonstration.
    In production, use a proper backtesting framework.
    """
    # Initialize metrics
    trades = []
    equity_curve = [strategy.initial_capital]
    
    # Generate signals
    trade_signals = strategy.generate_signals(market_data, signals)
    
    # Simulate trading
    current_position = None
    cash = strategy.initial_capital
    
    for i in range(1, len(market_data)):
        date = market_data.index[i]
        price = market_data.loc[date, 'close']
        signal = trade_signals.loc[date, 'signal'] if date in trade_signals.index else 0
        
        # Check for entry
        if signal > 0 and current_position is None:
            # Buy
            shares = (cash * strategy.position_size) / price
            cost = shares * price * (1 + strategy.commission)
            if cost <= cash:
                current_position = {
                    'entry_date': date,
                    'entry_price': price,
                    'shares': shares,
                    'cost': cost
                }
                cash -= cost
        
        # Check for exit
        elif signal < 0 and current_position is not None:
            # Sell
            proceeds = current_position['shares'] * price * (1 - strategy.commission)
            profit = proceeds - current_position['cost']
            
            trades.append({
                'entry_date': current_position['entry_date'],
                'exit_date': date,
                'entry_price': current_position['entry_price'],
                'exit_price': price,
                'shares': current_position['shares'],
                'profit': profit,
                'return': profit / current_position['cost']
            })
            
            cash += proceeds
            current_position = None
        
        # Update equity
        position_value = current_position['shares'] * price if current_position else 0
        total_equity = cash + position_value
        equity_curve.append(total_equity)
    
    # Calculate metrics
    equity_series = pd.Series(equity_curve[1:], index=market_data.index[1:])
    returns = equity_series.pct_change().dropna()
    
    metrics = {
        'Total Return': (equity_series.iloc[-1] / strategy.initial_capital - 1) * 100,
        'Annual Return': ((equity_series.iloc[-1] / strategy.initial_capital) ** (252 / len(returns)) - 1) * 100,
        'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
        'Max Drawdown': calculate_max_drawdown(equity_series) * 100,
        'Win Rate': len([t for t in trades if t['profit'] > 0]) / len(trades) * 100 if trades else 0,
        'Total Trades': len(trades),
        'Avg Trade Return': np.mean([t['return'] for t in trades]) * 100 if trades else 0,
        'Best Trade': max([t['return'] for t in trades]) * 100 if trades else 0,
        'Worst Trade': min([t['return'] for t in trades]) * 100 if trades else 0,
        'Profit Factor': calculate_profit_factor(trades)
    }
    
    return metrics


def calculate_max_drawdown(equity_series: pd.Series) -> float:
    """Calculate maximum drawdown."""
    cumulative = (1 + equity_series.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_profit_factor(trades: List[Dict]) -> float:
    """Calculate profit factor (gross profits / gross losses)."""
    if not trades:
        return 0
    
    gross_profits = sum(t['profit'] for t in trades if t['profit'] > 0)
    gross_losses = abs(sum(t['profit'] for t in trades if t['profit'] < 0))
    
    return gross_profits / gross_losses if gross_losses > 0 else float('inf')


def display_comparison_results(comparison_df: pd.DataFrame):
    """Display comparison results in a formatted way."""
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON RESULTS")
    print("=" * 100)
    
    # Sort by total return
    comparison_df = comparison_df.sort_values('Total Return', ascending=False)
    
    # Display main metrics
    display_columns = [
        'Strategy', 'Type', 'Total Return', 'Annual Return', 
        'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Total Trades'
    ]
    
    print("\nPerformance Summary:")
    print(comparison_df[display_columns].to_string(index=False, float_format='%.2f'))
    
    # Highlight best performers
    print("\n📊 Key Insights:")
    
    user_strategy = comparison_df[comparison_df['Type'] == 'User'].iloc[0]
    
    # Compare to Buy & Hold
    buy_hold = comparison_df[comparison_df['Strategy'].str.contains('Buy and Hold')]
    if not buy_hold.empty:
        bh_return = buy_hold.iloc[0]['Total Return']
        user_return = user_strategy['Total Return']
        
        if user_return > bh_return:
            print(f"✅ Strategy BEATS Buy & Hold by {user_return - bh_return:.1f}%")
        else:
            print(f"❌ Strategy UNDERPERFORMS Buy & Hold by {bh_return - user_return:.1f}%")
    
    # Best risk-adjusted return
    best_sharpe = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax()]
    print(f"\n🏆 Best Risk-Adjusted Return: {best_sharpe['Strategy']} (Sharpe: {best_sharpe['Sharpe Ratio']:.2f})")
    
    # Lowest drawdown
    best_dd = comparison_df.loc[comparison_df['Max Drawdown'].idxmin()]
    print(f"🛡️ Lowest Drawdown: {best_dd['Strategy']} ({best_dd['Max Drawdown']:.1f}%)")
    
    # Most active
    most_active = comparison_df.loc[comparison_df['Total Trades'].idxmax()]
    print(f"⚡ Most Active: {most_active['Strategy']} ({most_active['Total Trades']} trades)")
    
    # User strategy ranking
    user_rank = comparison_df.index.get_loc(comparison_df[comparison_df['Type'] == 'User'].index[0]) + 1
    print(f"\n📈 Your Strategy Rank: {user_rank} out of {len(comparison_df)}")


def generate_sample_data(symbols: List[str], days: int = 252) -> pd.DataFrame:
    """Generate sample market data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    data = {}
    
    for symbol in symbols:
        # Random walk with trend
        returns = np.random.normal(0.0005, 0.02, days)  # 0.05% daily return, 2% volatility
        prices = 100 * (1 + returns).cumprod()
        
        data[symbol] = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.002, days)),
            'high': prices * (1 + abs(np.random.normal(0, 0.005, days))),
            'low': prices * (1 - abs(np.random.normal(0, 0.005, days))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, days)
        }, index=dates)
    
    return data


def main():
    """Run benchmark comparison example."""
    print("🚀 TETRA Strategy Benchmark Comparison Tool")
    
    # Create a sample user strategy (you would replace this with your actual strategy)
    from src.strats.signal_based import SignalBasedStrategy, SignalRule, SignalCondition, ConditionOperator, PositionSide
    
    user_rules = [
        SignalRule(
            name="my_custom_rule",
            entry_conditions=[
                SignalCondition("rsi_14", ConditionOperator.LESS_THAN, 35),
                SignalCondition("close", ConditionOperator.GREATER_THAN, "sma_50")
            ],
            exit_conditions=[
                SignalCondition("rsi_14", ConditionOperator.GREATER_THAN, 65)
            ],
            position_side=PositionSide.LONG,
            stop_loss=0.05
        )
    ]
    
    user_strategy = SignalBasedStrategy(
        name="My Custom Strategy",
        signal_rules=user_rules,
        initial_capital=100000,
        position_size=0.1,
        max_positions=5
    )
    
    # Get benchmark strategies
    print("\n📊 Loading benchmark strategies...")
    benchmarks = get_core_benchmarks()
    
    print(f"Loaded {len(benchmarks)} benchmark strategies:")
    for name, strategy in benchmarks.items():
        print(f"  • {strategy.name}")
    
    # Generate sample data (replace with real data in production)
    print("\n📈 Generating sample market data...")
    sample_data = generate_sample_data(['SPY'], days=252)
    market_data = sample_data['SPY']
    
    # Run comparison
    print("\n🔄 Running backtests...")
    comparison_results = compare_strategies(user_strategy, benchmarks, market_data)
    
    # Display results
    display_comparison_results(comparison_results)
    
    # Additional analysis by style
    print("\n\n📊 ANALYSIS BY TRADING STYLE")
    print("=" * 60)
    
    styles = get_benchmarks_by_style()
    for style, strategy_names in styles.items():
        print(f"\n{style.upper().replace('_', ' ')}:")
        style_strategies = {name: get_benchmark_strategy(name) for name in strategy_names[:2]}  # Limit to 2 per style
        style_results = compare_strategies(user_strategy, style_strategies, market_data)
        
        # Show summary
        avg_return = style_results[style_results['Type'] == 'Benchmark']['Total Return'].mean()
        print(f"  Average Return: {avg_return:.1f}%")
        
        if user_strategy.name in style_results['Strategy'].values:
            user_return = style_results[style_results['Strategy'] == user_strategy.name]['Total Return'].iloc[0]
            if user_return > avg_return:
                print(f"  ✅ Your strategy beats this style average by {user_return - avg_return:.1f}%")
            else:
                print(f"  ❌ Your strategy underperforms this style by {avg_return - user_return:.1f}%")
    
    print("\n" + "=" * 100)
    print("✅ Benchmark comparison complete!")
    print("\nNext steps:")
    print("1. Review which benchmarks your strategy outperforms")
    print("2. Analyze why certain benchmarks perform better")
    print("3. Consider incorporating successful elements from top benchmarks")
    print("4. Test on different market conditions and time periods")


if __name__ == "__main__":
    main()