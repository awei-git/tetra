"""Run all strategy tests and create a comprehensive report."""

import logging
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.backtesting.engine import BacktestEngine, BacktestConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_strategy_tests(test_period: str = "short"):
    """Run all available strategy tests.
    
    Args:
        test_period: "short" (3 months) or "long" (6 months)
    """
    
    # Configure test period
    if test_period == "short":
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 31),
            initial_capital=100000,
            commission=0.001,
            slippage=0.0001,
            max_positions=5,
            calculate_metrics_every=10,
            benchmark=None
        )
    else:
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 30),
            initial_capital=100000,
            commission=0.001,
            slippage=0.0001,
            max_positions=10,
            calculate_metrics_every=20,
            benchmark=None
        )
    
    # Import strategies
    from test_simple_strategies import (
        SimpleTrendFollowing, SimpleRangeTrading, 
        SimpleDollarCostAverage, SimpleRotation
    )
    
    # Define test universe
    if test_period == "short":
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    else:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
                  'NVDA', 'TSLA', 'JPM', 'JNJ', 'WMT']
    
    # All strategies to test
    all_strategies = [
        # Simple strategies
        (SimpleTrendFollowing, "Trend Following (5d)", {'trend_days': 5}),
        (SimpleTrendFollowing, "Trend Following (10d)", {'trend_days': 10}),
        (SimpleRangeTrading, "Range Trading", {}),
        (SimpleDollarCostAverage, "DCA (Monthly)", {'buy_day': 1}),
        (SimpleDollarCostAverage, "DCA (Bi-weekly)", {'buy_day': 15}),
        (SimpleRotation, "Rotation (Top 2)", {'hold_days': 30, 'top_n': 2}),
        (SimpleRotation, "Rotation (Top 3)", {'hold_days': 20, 'top_n': 3}),
    ]
    
    # Run tests
    results = []
    
    print(f"\n{'='*100}")
    print(f"RUNNING STRATEGY TESTS - {test_period.upper()} PERIOD")
    print(f"Test Period: {config.start_date.date()} to {config.end_date.date()}")
    print(f"Universe: {len(symbols)} symbols")
    print(f"{'='*100}\n")
    
    for i, (strategy_class, name, kwargs) in enumerate(all_strategies, 1):
        print(f"[{i}/{len(all_strategies)}] Testing {name}...")
        
        engine = BacktestEngine(config=config)
        
        try:
            if kwargs:
                report = engine.run(
                    strategy=lambda *args, **kw: strategy_class(*args, **kwargs, **kw),
                    symbols=symbols,
                    signal_computer=None
                )
            else:
                report = engine.run(
                    strategy=strategy_class,
                    symbols=symbols,
                    signal_computer=None
                )
            
            results.append({
                'strategy': name,
                'total_return': report.total_return,
                'annualized_return': report.annualized_return,
                'volatility': report.volatility,
                'sharpe_ratio': report.sharpe_ratio,
                'max_drawdown': report.max_drawdown,
                'total_trades': report.total_trades,
                'win_rate': report.win_rate,
                'profit_factor': report.profit_factor,
                'avg_win': report.avg_win,
                'avg_loss': report.avg_loss,
                'final_equity': report.final_equity
            })
            
            print(f"   ✓ Return: {report.total_return:.2%}, Sharpe: {report.sharpe_ratio:.2f}")
            
        except Exception as e:
            logger.error(f"   ✗ Failed: {e}")
            results.append({
                'strategy': name,
                'error': str(e)
            })
    
    return results, config


def create_summary_report(results: List[Dict], config: BacktestConfig):
    """Create a comprehensive summary report."""
    
    print(f"\n{'='*120}")
    print("COMPREHENSIVE STRATEGY PERFORMANCE REPORT")
    print(f"{'='*120}")
    print(f"Test Period: {config.start_date.date()} to {config.end_date.date()}")
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Commission: {config.commission:.2%} | Slippage: {config.slippage:.2%}")
    print(f"{'='*120}\n")
    
    # Filter successful results
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    if successful:
        # Create DataFrame for easier analysis
        df = pd.DataFrame(successful)
        
        # Detailed results table
        print("DETAILED RESULTS:")
        print(f"{'-'*120}")
        print(f"{'Strategy':<25} {'Return':>8} {'Annual':>8} {'Vol':>8} {'Sharpe':>8} "
              f"{'MaxDD':>8} {'Trades':>8} {'WinRate':>8} {'PF':>8}")
        print(f"{'-'*120}")
        
        for _, row in df.iterrows():
            sharpe = f"{row['sharpe_ratio']:>8.2f}" if row['sharpe_ratio'] is not None else "     N/A"
            pf = f"{row['profit_factor']:>8.2f}" if row['profit_factor'] is not None else "     N/A"
            
            print(f"{row['strategy']:<25} {row['total_return']:>7.2%} "
                  f"{row['annualized_return']:>7.2%} {row['volatility']:>7.2%} {sharpe} "
                  f"{row['max_drawdown']:>7.2%} {row['total_trades']:>8d} "
                  f"{row['win_rate']:>7.2%} {pf}")
        
        print(f"\n{'='*120}")
        print("PERFORMANCE RANKINGS:")
        print(f"{'-'*120}")
        
        # Best by different metrics
        if len(df) > 0:
            # Total Return
            best_return = df.loc[df['total_return'].idxmax()]
            print(f"Best Return: {best_return['strategy']} ({best_return['total_return']:.2%})")
            
            # Risk-adjusted (Sharpe)
            df_sharpe = df.dropna(subset=['sharpe_ratio'])
            if len(df_sharpe) > 0:
                best_sharpe = df_sharpe.loc[df_sharpe['sharpe_ratio'].idxmax()]
                print(f"Best Risk-Adjusted: {best_sharpe['strategy']} (Sharpe: {best_sharpe['sharpe_ratio']:.2f})")
            
            # Lowest Drawdown
            best_dd = df.loc[df['max_drawdown'].idxmin()]
            print(f"Lowest Drawdown: {best_dd['strategy']} ({best_dd['max_drawdown']:.2%})")
            
            # Most Active
            most_active = df.loc[df['total_trades'].idxmax()]
            print(f"Most Active: {most_active['strategy']} ({most_active['total_trades']} trades)")
            
            # Best Win Rate (with min trades)
            active_strategies = df[df['total_trades'] >= 5]
            if len(active_strategies) > 0:
                best_wr = active_strategies.loc[active_strategies['win_rate'].idxmax()]
                print(f"Best Win Rate: {best_wr['strategy']} ({best_wr['win_rate']:.2%})")
        
        print(f"\n{'='*120}")
        print("RISK ANALYSIS:")
        print(f"{'-'*120}")
        
        # Risk metrics
        avg_return = df['total_return'].mean()
        avg_vol = df['volatility'].mean()
        avg_dd = df['max_drawdown'].mean()
        
        print(f"Average Return: {avg_return:.2%}")
        print(f"Average Volatility: {avg_vol:.2%}")
        print(f"Average Max Drawdown: {avg_dd:.2%}")
        
        # Positive returns
        positive_returns = len(df[df['total_return'] > 0])
        print(f"Strategies with Positive Returns: {positive_returns}/{len(df)} ({positive_returns/len(df)*100:.1f}%)")
        
        # Risk categories
        low_risk = df[df['volatility'] < avg_vol]
        high_return = df[df['total_return'] > avg_return]
        efficient = df[(df['total_return'] > avg_return) & (df['volatility'] < avg_vol)]
        
        print(f"Low Risk Strategies: {len(low_risk)}")
        print(f"High Return Strategies: {len(high_return)}")
        print(f"Efficient Strategies (High Return + Low Risk): {len(efficient)}")
        
        if len(efficient) > 0:
            print("\nEfficient Strategies:")
            for _, row in efficient.iterrows():
                print(f"  - {row['strategy']}: {row['total_return']:.2%} return, {row['volatility']:.2%} volatility")
    
    # Failed strategies
    if failed:
        print(f"\n{'='*120}")
        print(f"FAILED STRATEGIES: {len(failed)}")
        print(f"{'-'*120}")
        for f in failed:
            print(f"- {f['strategy']}: {f['error']}")
    
    print(f"\n{'='*120}")
    print(f"Test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*120}\n")


def main():
    """Run all tests and generate reports."""
    
    logger.info("Starting comprehensive strategy testing suite...")
    
    # Run short-term tests
    print("\n" + "="*50)
    print("SHORT-TERM TEST (3 months)")
    print("="*50)
    short_results, short_config = run_strategy_tests("short")
    
    # Run long-term tests
    print("\n" + "="*50)
    print("LONG-TERM TEST (6 months)")
    print("="*50)
    long_results, long_config = run_strategy_tests("long")
    
    # Generate reports
    print("\n\n" + "#"*120)
    print("SHORT-TERM RESULTS")
    print("#"*120)
    create_summary_report(short_results, short_config)
    
    print("\n\n" + "#"*120)
    print("LONG-TERM RESULTS")
    print("#"*120)
    create_summary_report(long_results, long_config)
    
    # Compare short vs long performance
    print("\n\n" + "#"*120)
    print("SHORT VS LONG TERM COMPARISON")
    print("#"*120)
    
    # Get successful strategies from both periods
    short_successful = {r['strategy']: r for r in short_results if 'error' not in r}
    long_successful = {r['strategy']: r for r in long_results if 'error' not in r}
    
    common_strategies = set(short_successful.keys()) & set(long_successful.keys())
    
    if common_strategies:
        print(f"\n{'Strategy':<25} {'Short Return':>12} {'Long Return':>12} {'Consistency':>12}")
        print("-"*65)
        
        for strategy in sorted(common_strategies):
            short_ret = short_successful[strategy]['total_return']
            long_ret = long_successful[strategy]['total_return']
            
            # Consistency score
            if short_ret > 0 and long_ret > 0:
                consistency = "Consistent +"
            elif short_ret < 0 and long_ret < 0:
                consistency = "Consistent -"
            else:
                consistency = "Mixed"
            
            print(f"{strategy:<25} {short_ret:>11.2%} {long_ret:>11.2%} {consistency:>12}")
    
    logger.info("Strategy testing completed!")


if __name__ == "__main__":
    main()