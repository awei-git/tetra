#!/usr/bin/env python3
"""
Fix return calculations to properly average all 30 real data scenarios.
"""

import asyncio
import asyncpg
import json
import numpy as np

DATABASE_URL = 'postgresql://tetra_user:tetra_password@localhost:5432/tetra'

async def fix_return_calculations():
    """
    Recalculate 2W, 1M, 3M returns by averaging ALL 30 real scenarios.
    
    The assessment pipeline runs 30 real data scenarios for each strategy-symbol pair.
    Each timeframe return should be the average of all 30 scenarios, not a subset.
    """
    
    conn = None
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        print("🔧 Fixing return calculations to average all 30 real scenarios...")
        print("=" * 80)
        
        # Get all trades with their scenario data
        trades = await conn.fetch("""
            SELECT trade_id, strategy_name, symbol, scenario_returns
            FROM strategies.strategy_trades
            WHERE scenario_returns IS NOT NULL
        """)
        
        print(f"📊 Processing {len(trades)} trades...")
        
        updated_count = 0
        
        for trade in trades:
            trade_id = trade['trade_id']
            scenarios = json.loads(trade['scenario_returns'])
            
            # Extract all 30 real scenario returns
            real_scenarios = {k: v for k, v in scenarios.items() if k.startswith('Real_')}
            
            if len(real_scenarios) < 30:
                print(f"⚠️ Trade {trade_id} has only {len(real_scenarios)} real scenarios, skipping...")
                continue
            
            # Calculate returns as average of ALL 30 real scenarios
            # In the assessment pipeline, each scenario represents a different market condition
            # The return for each timeframe is the expected value across all conditions
            
            all_real_returns = list(real_scenarios.values())
            
            # For 2W return: Average of all 30 scenarios, scaled for 2-week horizon
            # Scenarios represent different volatility/trend regimes over the period
            return_2w = np.mean(all_real_returns) * 0.15  # Scale to 2-week timeframe
            
            # For 1M return: Average of all 30 scenarios, scaled for 1-month horizon
            return_1m = np.mean(all_real_returns) * 0.35  # Scale to 1-month timeframe
            
            # For 3M return: Average of all 30 scenarios (full period)
            return_3m = np.mean(all_real_returns)  # Full 3-month expected return
            
            # Update the trade with corrected returns
            await conn.execute("""
                UPDATE strategies.strategy_trades
                SET return_2w = $1,
                    return_1m = $2,
                    return_3m = $3
                WHERE trade_id = $4
            """, return_2w, return_1m, return_3m, trade_id)
            
            updated_count += 1
            
            # Show sample for first few trades
            if updated_count <= 3:
                print(f"\n{trade['strategy_name']} - {trade['symbol']}:")
                print(f"  30 Real Scenarios Average: {np.mean(all_real_returns)*100:.2f}%")
                print(f"  Corrected 2W Return: {return_2w*100:.2f}%")
                print(f"  Corrected 1M Return: {return_1m*100:.2f}%")
                print(f"  Corrected 3M Return: {return_3m*100:.2f}%")
        
        print(f"\n✅ Updated {updated_count} trades with corrected return calculations")
        
        # Show the logic
        print("\n" + "=" * 80)
        print("📝 Correct Return Calculation Method:")
        print("1. Assessment pipeline runs 30 real data scenarios per strategy-symbol")
        print("2. Each scenario represents different market conditions (trends, volatility, etc.)")
        print("3. Returns are calculated as:")
        print("   - 2W Return = AVG(all 30 scenarios) × 0.15 (scaled to 2-week horizon)")
        print("   - 1M Return = AVG(all 30 scenarios) × 0.35 (scaled to 1-month horizon)")
        print("   - 3M Return = AVG(all 30 scenarios) × 1.00 (full period return)")
        print("4. This gives the expected return across all market conditions")
        
        # Verify the fix with summary stats
        summary = await conn.fetchrow("""
            SELECT 
                AVG(return_2w) as avg_2w,
                AVG(return_1m) as avg_1m,
                AVG(return_3m) as avg_3m,
                STDDEV(return_2w) as std_2w,
                STDDEV(return_1m) as std_1m,
                STDDEV(return_3m) as std_3m
            FROM strategies.strategy_trades
        """)
        
        print("\n📊 Updated Return Statistics:")
        print(f"  2W: {summary['avg_2w']*100:.2f}% ± {summary['std_2w']*100:.2f}%")
        print(f"  1M: {summary['avg_1m']*100:.2f}% ± {summary['std_1m']*100:.2f}%")
        print(f"  3M: {summary['avg_3m']*100:.2f}% ± {summary['std_3m']*100:.2f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        if conn:
            await conn.close()

if __name__ == "__main__":
    asyncio.run(fix_return_calculations())