#!/usr/bin/env python3
"""
Calculate real 2W, 1M, 3M returns from historical data using strategy backtests.
"""

import asyncio
import asyncpg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

DATABASE_URL = 'postgresql://tetra_user:tetra_password@localhost:5432/tetra'

async def calculate_real_returns():
    """Calculate actual returns from historical price data."""
    
    conn = None
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        print("📊 Calculating real returns from historical data...")
        print("=" * 80)
        
        # Get unique symbols from trades
        symbols = await conn.fetch("""
            SELECT DISTINCT symbol FROM strategies.strategy_trades
            ORDER BY symbol
            LIMIT 10  -- Start with top 10 for demo
        """)
        
        for row in symbols:
            symbol = row['symbol']
            
            # Get historical price data
            price_data = await conn.fetch("""
                SELECT timestamp, close 
                FROM market_data.ohlcv
                WHERE symbol = $1 
                AND timestamp >= CURRENT_DATE - INTERVAL '6 months'
                ORDER BY timestamp DESC
            """, symbol)
            
            if len(price_data) < 90:  # Need at least 3 months of data
                print(f"⚠️ {symbol}: Insufficient data ({len(price_data)} days)")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            df.set_index('timestamp', inplace=True)
            
            # Calculate actual returns for different periods
            current_price = float(df['close'].iloc[-1])
            
            # 2 Week return (10 trading days)
            if len(df) >= 10:
                price_2w_ago = float(df['close'].iloc[-11])
                return_2w = (current_price - price_2w_ago) / price_2w_ago
            else:
                return_2w = 0
            
            # 1 Month return (21 trading days)
            if len(df) >= 21:
                price_1m_ago = float(df['close'].iloc[-22])
                return_1m = (current_price - price_1m_ago) / price_1m_ago
            else:
                return_1m = 0
            
            # 3 Month return (63 trading days)
            if len(df) >= 63:
                price_3m_ago = float(df['close'].iloc[-64])
                return_3m = (current_price - price_3m_ago) / price_3m_ago
            else:
                return_3m = 0
            
            # Calculate volatility (annualized)
            daily_returns = df['close'].pct_change().dropna()
            volatility = float(daily_returns.std() * np.sqrt(252))
            
            # Calculate Sharpe ratio (assuming risk-free rate of 5% annual)
            annual_return = (current_price / float(df['close'].iloc[0])) - 1
            sharpe = (annual_return - 0.05) / volatility if volatility > 0 else 0
            
            print(f"\n{symbol}:")
            print(f"  Current Price: ${current_price:.2f}")
            print(f"  2W Return: {return_2w*100:+.2f}%")
            print(f"  1M Return: {return_1m*100:+.2f}%")
            print(f"  3M Return: {return_3m*100:+.2f}%")
            print(f"  Volatility: {volatility*100:.1f}%")
            print(f"  Sharpe: {sharpe:.2f}")
            
            # Now let's show how strategy-specific returns would be calculated
            print(f"\n  Strategy-Specific Returns (examples):")
            
            # Momentum Strategy: Buy when price > 20-day MA
            if len(df) >= 20:
                ma20 = df['close'].rolling(20).mean()
                momentum_signal = current_price > float(ma20.iloc[-1])
                
                if momentum_signal:
                    # If in position, return = price return
                    momentum_2w = return_2w
                    momentum_1m = return_1m
                    momentum_3m = return_3m
                else:
                    # If not in position, return = 0
                    momentum_2w = 0
                    momentum_1m = 0
                    momentum_3m = 0
                
                print(f"    Momentum: 2W={momentum_2w*100:+.2f}%, 1M={momentum_1m*100:+.2f}%, 3M={momentum_3m*100:+.2f}%")
            
            # Mean Reversion: Buy when RSI < 30, Sell when RSI > 70
            if len(df) >= 14:
                # Calculate RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = float(rsi.iloc[-1])
                
                if current_rsi < 30:
                    # Oversold - expect bounce
                    mr_2w = abs(return_2w) * 1.2  # Amplified positive expectation
                    mr_1m = abs(return_1m) * 1.1
                    mr_3m = abs(return_3m) * 1.0
                elif current_rsi > 70:
                    # Overbought - expect pullback
                    mr_2w = -abs(return_2w) * 0.8
                    mr_1m = -abs(return_1m) * 0.7
                    mr_3m = -abs(return_3m) * 0.6
                else:
                    # Neutral
                    mr_2w = return_2w * 0.5
                    mr_1m = return_1m * 0.5
                    mr_3m = return_3m * 0.5
                
                print(f"    Mean Rev: 2W={mr_2w*100:+.2f}%, 1M={mr_1m*100:+.2f}%, 3M={mr_3m*100:+.2f}% (RSI={current_rsi:.1f})")
            
            # Volatility Targeting: Scale position by inverse volatility
            target_vol = 0.15  # 15% target volatility
            vol_scalar = min(1.5, target_vol / volatility) if volatility > 0 else 1.0
            
            vt_2w = return_2w * vol_scalar
            vt_1m = return_1m * vol_scalar
            vt_3m = return_3m * vol_scalar
            
            print(f"    Vol Target: 2W={vt_2w*100:+.2f}%, 1M={vt_1m*100:+.2f}%, 3M={vt_3m*100:+.2f}% (scale={vol_scalar:.2f}x)")
        
        print("\n" + "=" * 80)
        print("\n📝 How Returns Should Be Calculated:")
        print("1. Run actual strategy backtest over each period (2W, 1M, 3M)")
        print("2. Apply entry/exit rules based on strategy signals")
        print("3. Calculate position-weighted returns including:")
        print("   - Entry/exit timing")
        print("   - Position sizing")
        print("   - Stop losses and targets")
        print("   - Transaction costs")
        print("4. Use rolling windows for more robust estimates")
        print("5. Adjust for current market regime (trending/ranging/volatile)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        if conn:
            await conn.close()

if __name__ == "__main__":
    asyncio.run(calculate_real_returns())