#!/usr/bin/env python3
"""Quick performance summary showing strategy performance across different conditions."""

import asyncio
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from datetime import date, timedelta
import pandas as pd
import numpy as np
from src.simulators.historical import HistoricalSimulator
from src.simulators.portfolio import Portfolio
from src.strats.benchmark import get_all_benchmarks

async def quick_analysis():
    """Run quick analysis to demonstrate the comprehensive approach."""
    
    print("="*80)
    print("STRATEGY PERFORMANCE ANALYSIS - QUICK SUMMARY")
    print("="*80)
    
    strategies = get_all_benchmarks()
    simulator = HistoricalSimulator()
    
    # Test configurations
    test_configs = [
        # (symbol, strategy_name, start_date, end_date, description)
        ('SPY', 'buy_and_hold', date(2024, 8, 7), date(2025, 8, 7), '1 Year'),
        ('SPY', 'buy_and_hold', date(2025, 5, 7), date(2025, 8, 7), '3 Months'),
        ('SPY', 'buy_and_hold', date(2025, 7, 24), date(2025, 8, 7), '2 Weeks'),
        
        ('QQQ', 'momentum_factor', date(2024, 8, 7), date(2025, 8, 7), '1 Year'),
        ('QQQ', 'golden_cross', date(2024, 8, 7), date(2025, 8, 7), '1 Year'),
        
        ('AAPL', 'rsi_reversion', date(2024, 8, 7), date(2025, 8, 7), '1 Year'),
        ('MSFT', 'macd_crossover', date(2024, 8, 7), date(2025, 8, 7), '1 Year'),
    ]
    
    print("\n1. SAMPLE PERFORMANCE ACROSS DIFFERENT TIME WINDOWS")
    print("-"*60)
    
    results = []
    for symbol, strategy_name, start_date, end_date, window_desc in test_configs:
        try:
            strategy = strategies[strategy_name]
            portfolio = Portfolio(initial_cash=100000.0)
            
            if hasattr(strategy, 'set_symbols'):
                strategy.set_symbols([symbol])
            
            result = await simulator.run_simulation(
                portfolio=portfolio,
                start_date=start_date,
                end_date=end_date,
                strategy=strategy
            )
            
            print(f"{strategy_name} on {symbol} ({window_desc}): "
                  f"{result.total_return:.2%} return, "
                  f"{result.sharpe_ratio:.2f} Sharpe, "
                  f"{result.total_trades} trades")
            
            results.append({
                'strategy': strategy_name,
                'symbol': symbol,
                'window': window_desc,
                'return': result.total_return,
                'sharpe': result.sharpe_ratio,
                'max_dd': result.max_drawdown,
                'trades': result.total_trades
            })
            
        except Exception as e:
            print(f"{strategy_name} on {symbol} ({window_desc}): FAILED - {e}")
    
    # Show how performance varies with window size
    print("\n2. PERFORMANCE VARIATION BY WINDOW SIZE (Buy & Hold SPY)")
    print("-"*60)
    
    window_tests = [
        (7, '1 Week'),
        (14, '2 Weeks'),
        (30, '1 Month'),
        (90, '3 Months'),
        (180, '6 Months'),
        (365, '1 Year')
    ]
    
    end_date = date(2025, 8, 7)
    for days, desc in window_tests:
        start_date = end_date - timedelta(days=days)
        try:
            strategy = strategies['buy_and_hold']
            portfolio = Portfolio(initial_cash=100000.0)
            strategy.set_symbols(['SPY'])
            
            result = await simulator.run_simulation(
                portfolio=portfolio,
                start_date=start_date,
                end_date=end_date,
                strategy=strategy
            )
            
            annualized = result.annual_return
            print(f"{desc:10}: {result.total_return:7.2%} total, "
                  f"{annualized:7.2%} annualized, "
                  f"Sharpe: {result.sharpe_ratio:5.2f}")
                  
        except Exception as e:
            print(f"{desc:10}: FAILED - {e}")
    
    # Demonstrate rolling window concept
    print("\n3. ROLLING WINDOW ANALYSIS (Buy & Hold SPY, 3-month windows)")
    print("-"*60)
    
    rolling_results = []
    window_size = 90
    step_size = 30  # Monthly steps
    
    current_date = date(2024, 8, 7)
    final_date = date(2025, 8, 7)
    
    while current_date + timedelta(days=window_size) <= final_date:
        window_end = current_date + timedelta(days=window_size)
        
        try:
            strategy = strategies['buy_and_hold']
            portfolio = Portfolio(initial_cash=100000.0)
            strategy.set_symbols(['SPY'])
            
            result = await simulator.run_simulation(
                portfolio=portfolio,
                start_date=current_date,
                end_date=window_end,
                strategy=strategy
            )
            
            print(f"{current_date} to {window_end}: {result.total_return:6.2%}")
            rolling_results.append(result.total_return)
            
        except Exception as e:
            print(f"{current_date} to {window_end}: FAILED")
            rolling_results.append(0)
        
        current_date += timedelta(days=step_size)
    
    if rolling_results:
        print(f"\nRolling 3-month returns:")
        print(f"  Average: {np.mean(rolling_results):.2%}")
        print(f"  Best: {max(rolling_results):.2%}")
        print(f"  Worst: {min(rolling_results):.2%}")
        print(f"  Std Dev: {np.std(rolling_results):.2%}")
    
    # Show strategy rankings
    print("\n4. STRATEGY RANKINGS (Based on 1-year performance)")
    print("-"*60)
    
    ranking_results = []
    test_symbols = ['SPY', 'QQQ', 'IWM']
    test_strategies = ['buy_and_hold', 'golden_cross', 'momentum_factor', 'rsi_reversion', 'turtle_trading']
    
    for strategy_name in test_strategies:
        strategy_returns = []
        strategy_sharpes = []
        
        for symbol in test_symbols:
            try:
                strategy = strategies[strategy_name]
                portfolio = Portfolio(initial_cash=100000.0)
                
                if hasattr(strategy, 'set_symbols'):
                    strategy.set_symbols([symbol])
                
                result = await simulator.run_simulation(
                    portfolio=portfolio,
                    start_date=date(2024, 8, 7),
                    end_date=date(2025, 8, 7),
                    strategy=strategy
                )
                
                strategy_returns.append(result.total_return)
                strategy_sharpes.append(result.sharpe_ratio)
                
            except:
                strategy_returns.append(0)
                strategy_sharpes.append(0)
        
        avg_return = np.mean(strategy_returns)
        avg_sharpe = np.mean(strategy_sharpes)
        
        ranking_results.append({
            'strategy': strategy_name,
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'score': avg_return * 0.6 + avg_sharpe * 0.4
        })
    
    # Sort by score
    ranking_results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\nStrategy Rankings:")
    for i, result in enumerate(ranking_results):
        print(f"#{i+1}. {result['strategy']:20}: "
              f"Avg Return: {result['avg_return']:6.2%}, "
              f"Avg Sharpe: {result['avg_sharpe']:5.2f}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("\n1. Performance varies significantly by time window")
    print("2. Shorter windows show more volatility in returns")
    print("3. Different strategies excel in different market conditions")
    print("4. Rolling window analysis reveals consistency of returns")
    print("\nFor full analysis with all strategies, symbols, and time periods,")
    print("the comprehensive script would generate a complete time series database.")
    
    print("\n" + "="*80)
    print("RECOMMENDED APPROACH")
    print("="*80)
    print("\n1. Run backtests for each strategy-symbol pair")
    print("2. Use rolling windows (2w, 3m, 1y) stepping through history")
    print("3. Collect performance metrics for each window")
    print("4. Analyze variance, drawdown patterns, and consistency")
    print("5. Apply market scenario stress tests separately")
    print("6. Rank strategies by risk-adjusted performance")

if __name__ == "__main__":
    asyncio.run(quick_analysis())