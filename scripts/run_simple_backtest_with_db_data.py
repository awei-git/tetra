#!/usr/bin/env python3
"""Run simple backtests using actual market data from the database."""

import asyncio
import asyncpg
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


async def get_market_data(symbol: str, start_date: datetime, end_date: datetime):
    """Get OHLCV data from database."""
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="tetra_user", 
        password="tetra_password",
        database="tetra"
    )
    
    try:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM market_data.ohlcv
            WHERE symbol = $1 
            AND timestamp >= $2
            AND timestamp <= $3
            AND timeframe = '1d'
            ORDER BY timestamp ASC
        """
        
        rows = await conn.fetch(query, symbol, start_date, end_date)
        
        if rows:
            df = pd.DataFrame(rows)
            df.set_index('timestamp', inplace=True)
            return df
        else:
            return None
            
    finally:
        await conn.close()


def calculate_strategy_performance(df: pd.DataFrame, strategy_type: str):
    """Calculate performance for different strategy types."""
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    if strategy_type == "buy_and_hold":
        # Buy on day 1, hold until end
        total_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
        trades = 1
        
    elif strategy_type == "golden_cross":
        # 50/200 SMA crossover
        df['sma50'] = df['close'].rolling(50).mean()
        df['sma200'] = df['close'].rolling(200).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['sma50'] > df['sma200'], 'signal'] = 1
        df['position'] = df['signal'].diff()
        
        # Calculate returns
        df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
        total_return = (1 + df['strategy_returns']).prod() - 1
        trades = abs(df['position']).sum()
        
    elif strategy_type == "mean_reversion":
        # RSI-based mean reversion
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Buy when RSI < 30, sell when RSI > 70
        df['signal'] = 0
        df.loc[df['rsi'] < 30, 'signal'] = 1
        df.loc[df['rsi'] > 70, 'signal'] = 0
        
        df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
        total_return = (1 + df['strategy_returns']).prod() - 1
        trades = abs(df['signal'].diff()).sum()
        
    elif strategy_type == "momentum":
        # Simple momentum - buy when 20-day return > 0
        df['momentum'] = df['close'].pct_change(20)
        df['signal'] = (df['momentum'] > 0.05).astype(int)
        
        df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
        total_return = (1 + df['strategy_returns']).prod() - 1
        trades = abs(df['signal'].diff()).sum()
        
    else:
        total_return = 0
        trades = 0
    
    # Calculate metrics
    if 'strategy_returns' in df:
        returns_series = df['strategy_returns'].dropna()
    else:
        returns_series = df['returns'].dropna()
        
    volatility = returns_series.std() * np.sqrt(252)
    sharpe_ratio = (returns_series.mean() * 252) / volatility if volatility > 0 else 0
    
    # Max drawdown
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trades': int(trades)
    }


async def run_strategy_analysis():
    """Analyze which stocks work best with each strategy."""
    
    # Define strategies and suitable stocks
    strategy_mapping = {
        "buy_and_hold": ["SPY", "AAPL", "MSFT", "BRK-B"],
        "golden_cross": ["AAPL", "MSFT", "GOOGL", "NVDA"],
        "mean_reversion": ["JPM", "BAC", "WFC", "GS"],
        "momentum": ["TSLA", "NVDA", "AMD", "NFLX"]
    }
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year
    
    print(f"Analyzing strategies from {start_date.date()} to {end_date.date()}")
    print("=" * 80)
    
    results = []
    
    for strategy, symbols in strategy_mapping.items():
        print(f"\n{strategy.upper()} STRATEGY")
        print("-" * 40)
        
        for symbol in symbols:
            df = await get_market_data(symbol, start_date, end_date)
            
            if df is not None and len(df) > 200:  # Need enough data
                perf = calculate_strategy_performance(df, strategy)
                
                results.append({
                    'strategy': strategy,
                    'symbol': symbol,
                    'return': perf['total_return'],
                    'sharpe': perf['sharpe_ratio'],
                    'max_dd': perf['max_drawdown'],
                    'volatility': perf['volatility'],
                    'trades': perf['trades']
                })
                
                print(f"{symbol:.<10} Return: {perf['total_return']:>7.1%}, "
                      f"Sharpe: {perf['sharpe_ratio']:>5.2f}, "
                      f"MaxDD: {perf['max_drawdown']:>6.1%}, "
                      f"Trades: {perf['trades']:>3}")
            else:
                print(f"{symbol:.<10} No data available")
    
    # Summary
    print("\n" + "=" * 80)
    print("BEST PERFORMING COMBINATIONS:")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Sort by Sharpe ratio
        top_results = results_df.nlargest(10, 'sharpe')
        
        for _, row in top_results.iterrows():
            print(f"{row['strategy']:.<20} + {row['symbol']:.<10} | "
                  f"Return: {row['return']:>7.1%}, Sharpe: {row['sharpe']:>5.2f}")
    
    # Save results to database
    await save_results_to_db(results)
    
    print("\n✅ Results saved to database. Check strategies tab!")


async def save_results_to_db(results):
    """Save backtest results to database."""
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="tetra_user",
        password="tetra_password",
        database="tetra"
    )
    
    try:
        run_date = datetime.now()
        
        for r in results:
            # Insert backtest result
            await conn.execute("""
                INSERT INTO strategies.backtest_results (
                    strategy_name, run_date, backtest_start_date, backtest_end_date,
                    universe, initial_capital, final_value, total_return,
                    annualized_return, sharpe_ratio, max_drawdown, volatility,
                    win_rate, total_trades, metadata, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """,
                f"{r['strategy']}_{r['symbol']}", run_date,
                run_date - timedelta(days=365), run_date,
                'single_stock', 100000,
                100000 * (1 + r['return']), r['return'],
                r['return'],  # Simplified - already annualized
                r['sharpe'], r['max_dd'], r['volatility'],
                0.5,  # Placeholder win rate
                r['trades'],
                '{}', run_date
            )
            
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_strategy_analysis())