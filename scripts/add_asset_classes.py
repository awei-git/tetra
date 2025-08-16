#!/usr/bin/env python3
"""
Add asset class categorization to trades.
"""

import asyncio
import asyncpg
import json

DATABASE_URL = 'postgresql://tetra_user:tetra_password@localhost:5432/tetra'

# Asset class mappings
ASSET_CLASSES = {
    # Equity ETFs
    'SPY': 'Equity ETF',
    'QQQ': 'Tech ETF', 
    'IWM': 'Small Cap ETF',
    'DIA': 'Equity ETF',
    'VOO': 'Equity ETF',
    'VTI': 'Equity ETF',
    'EFA': 'International ETF',
    'EEM': 'Emerging Markets ETF',
    
    # Sector ETFs
    'XLF': 'Financial Sector',
    'XLK': 'Tech Sector',
    'XLE': 'Energy Sector',
    'XLV': 'Healthcare Sector',
    'XLI': 'Industrial Sector',
    'XLY': 'Consumer Disc.',
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities Sector',
    'XLRE': 'Real Estate',
    'XLB': 'Materials Sector',
    'XLC': 'Communication',
    
    # Individual Stocks
    'AAPL': 'Tech Stock',
    'MSFT': 'Tech Stock',
    'GOOGL': 'Tech Stock',
    'AMZN': 'Tech Stock',
    'META': 'Tech Stock',
    'NVDA': 'Tech Stock',
    'TSLA': 'Auto/Tech Stock',
    'AMD': 'Tech Stock',
    'INTC': 'Tech Stock',
    'NFLX': 'Tech Stock',
    'ADBE': 'Tech Stock',
    'CRM': 'Tech Stock',
    'ORCL': 'Tech Stock',
    'CSCO': 'Tech Stock',
    'AVGO': 'Tech Stock',
    'QCOM': 'Tech Stock',
    'TXN': 'Tech Stock',
    'MU': 'Tech Stock',
    'AMAT': 'Tech Stock',
    'LRCX': 'Tech Stock',
    'KLAC': 'Tech Stock',
    'ASML': 'Tech Stock',
    'TSM': 'Tech Stock',
    
    # Financial Stocks
    'JPM': 'Financial Stock',
    'BAC': 'Financial Stock',
    'WFC': 'Financial Stock',
    'GS': 'Financial Stock',
    'MS': 'Financial Stock',
    'C': 'Financial Stock',
    'BLK': 'Financial Stock',
    'SCHW': 'Financial Stock',
    'AXP': 'Financial Stock',
    'V': 'Financial Stock',
    'MA': 'Financial Stock',
    'PYPL': 'Financial Stock',
    'SQ': 'Financial Stock',
    
    # Healthcare Stocks
    'JNJ': 'Healthcare Stock',
    'UNH': 'Healthcare Stock',
    'PFE': 'Healthcare Stock',
    'ABBV': 'Healthcare Stock',
    'MRK': 'Healthcare Stock',
    'TMO': 'Healthcare Stock',
    'ABT': 'Healthcare Stock',
    'DHR': 'Healthcare Stock',
    'LLY': 'Healthcare Stock',
    'CVS': 'Healthcare Stock',
    
    # Consumer Stocks
    'WMT': 'Consumer Stock',
    'HD': 'Consumer Stock',
    'PG': 'Consumer Stock',
    'KO': 'Consumer Stock',
    'PEP': 'Consumer Stock',
    'MCD': 'Consumer Stock',
    'NKE': 'Consumer Stock',
    'SBUX': 'Consumer Stock',
    'DIS': 'Entertainment Stock',
    'COST': 'Consumer Stock',
    'TGT': 'Consumer Stock',
    
    # Energy Stocks
    'XOM': 'Energy Stock',
    'CVX': 'Energy Stock',
    'COP': 'Energy Stock',
    'SLB': 'Energy Stock',
    'EOG': 'Energy Stock',
    'MPC': 'Energy Stock',
    'PSX': 'Energy Stock',
    'VLO': 'Energy Stock',
    
    # Commodity ETFs
    'GLD': 'Gold ETF',
    'SLV': 'Silver ETF',
    'USO': 'Oil ETF',
    'UNG': 'Natural Gas ETF',
    'DBA': 'Agriculture ETF',
    'PDBC': 'Commodity ETF',
    
    # Bond ETFs
    'TLT': 'Treasury Bond ETF',
    'IEF': 'Treasury Bond ETF',
    'SHY': 'Treasury Bond ETF',
    'AGG': 'Bond ETF',
    'BND': 'Bond ETF',
    'LQD': 'Corporate Bond ETF',
    'HYG': 'High Yield Bond',
    'EMB': 'EM Bond ETF',
    'TIP': 'TIPS ETF',
    
    # Crypto/Alternative
    'GBTC': 'Crypto Trust',
    'BITO': 'Bitcoin ETF',
    'COIN': 'Crypto Stock',
    'MARA': 'Crypto Mining',
    'RIOT': 'Crypto Mining',
    
    # Volatility
    'VXX': 'Volatility ETF',
    'UVXY': 'Volatility ETF',
    'VIXY': 'Volatility ETF',
    'VIX': 'Volatility Index',
    
    # International
    'FXI': 'China ETF',
    'EWJ': 'Japan ETF',
    'EWZ': 'Brazil ETF',
    'EWT': 'Taiwan ETF',
    'EWY': 'South Korea ETF',
    'INDA': 'India ETF',
    'RSX': 'Russia ETF',
    'EWU': 'UK ETF',
    'EWG': 'Germany ETF',
    
    # ARK ETFs
    'ARKK': 'Innovation ETF',
    'ARKQ': 'Automation ETF',
    'ARKW': 'Internet ETF',
    'ARKG': 'Genomics ETF',
    'ARKF': 'Fintech ETF',
    
    # Other
    'SQQQ': 'Inverse ETF',
    'TQQQ': 'Leveraged ETF',
    'SPXU': 'Inverse ETF',
    'SPXL': 'Leveraged ETF'
}

async def update_asset_classes():
    """Update trades with asset class information."""
    
    conn = None
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Get all unique symbols from trades
        symbols = await conn.fetch("""
            SELECT DISTINCT symbol FROM strategies.strategy_trades
        """)
        
        print(f"📊 Found {len(symbols)} unique symbols in trades")
        
        # Update each symbol with its asset class
        updated = 0
        for row in symbols:
            symbol = row['symbol']
            asset_class = ASSET_CLASSES.get(symbol, 'Equity Stock')  # Default to Equity Stock
            
            await conn.execute("""
                UPDATE strategies.strategy_trades 
                SET asset_class = $1 
                WHERE symbol = $2
            """, asset_class, symbol)
            
            updated += 1
            print(f"  ✓ {symbol}: {asset_class}")
        
        print(f"\n✅ Updated {updated} symbols with asset classes")
        
        # Show summary by asset class
        summary = await conn.fetch("""
            SELECT asset_class, COUNT(*) as count, 
                   AVG(composite_score) as avg_score,
                   AVG(expected_return) as avg_return
            FROM strategies.strategy_trades
            GROUP BY asset_class
            ORDER BY avg_score DESC
        """)
        
        print("\n📈 Asset Class Summary:")
        print("-" * 60)
        for row in summary:
            print(f"{row['asset_class']:20} | Count: {row['count']:3} | Avg Score: {row['avg_score']:.1f} | Avg Return: {row['avg_return']*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        if conn:
            await conn.close()

if __name__ == "__main__":
    print("🚀 Adding asset class categorization to trades...")
    print("=" * 80)
    asyncio.run(update_asset_classes())