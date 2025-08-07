#!/usr/bin/env python3
"""
Update stale market data and economic indicators
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.data_ingester import DataIngester
from src.pipelines.data_pipeline.steps.economic_data import EconomicDataStep
from src.pipelines.base import PipelineContext
from src.utils.logging import logger


async def update_stale_market_data():
    """Update stale market data symbols"""
    logger.info("=== Updating Stale Market Data ===")
    
    # Stale symbols that need full historical update
    stale_symbols = {
        'VIIX': datetime(2023, 10, 31).date(),  # Last update 642 days ago
        'MATIC-USD': datetime(2025, 3, 24).date()  # Last update 132 days ago
    }
    
    # Crypto symbols that are 1 day behind
    crypto_symbols = [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
        'SOL-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD'
    ]
    
    ingester = DataIngester(provider="polygon")
    total_records = 0
    
    # Update stale symbols with full history
    for symbol, last_date in stale_symbols.items():
        logger.info(f"\nUpdating {symbol} from {last_date} to present...")
        try:
            # For VIIX, it might be delisted, try yfinance
            if symbol == 'VIIX':
                logger.info(f"{symbol} might be delisted, trying yfinance...")
                yf_ingester = DataIngester(provider="yfinance")
                result = await yf_ingester.ingest_ohlcv_batch(
                    symbols=[symbol],
                    from_date=last_date,
                    to_date=datetime.now().date(),
                    timeframe="1d"
                )
            else:
                result = await ingester.ingest_ohlcv_batch(
                    symbols=[symbol],
                    from_date=last_date,
                    to_date=datetime.now().date(),
                    timeframe="1d"
                )
            
            records = result.get("total_records", 0)
            total_records += records
            logger.info(f"Updated {symbol}: {records} new records")
            
        except Exception as e:
            logger.error(f"Failed to update {symbol}: {e}")
    
    # Update crypto (just 1-2 days behind)
    logger.info(f"\nUpdating {len(crypto_symbols)} crypto symbols...")
    try:
        result = await ingester.ingest_ohlcv_batch(
            symbols=crypto_symbols,
            from_date=datetime.now().date() - timedelta(days=7),  # Get last week to be safe
            to_date=datetime.now().date(),
            timeframe="1d"
        )
        
        records = result.get("total_records", 0)
        total_records += records
        logger.info(f"Updated crypto: {records} new records")
        
    except Exception as e:
        logger.error(f"Failed to update crypto: {e}")
    
    logger.info(f"\nTotal market data records updated: {total_records}")
    return total_records


async def update_stale_economic_data():
    """Update stale economic indicators"""
    logger.info("\n=== Updating Stale Economic Data ===")
    
    # These indicators need updates
    stale_indicators = [
        'GDPC1',      # GDP - 124 days old
        'M2V',        # Money velocity - 124 days old
        'A191RL1Q225SBEA',  # Real GDP growth - 124 days old
        'UMCSENT',    # Consumer sentiment - 94 days old
        'BUSINV',     # Business inventories - 94 days old
        'CSUSHPISA',  # Case-Shiller home price index - 94 days old
        'TEDRATE',    # TED spread - very old
        'GFDEBTN',    # Federal debt - 214 days old
        'CPIAUCSL',   # CPI - needs update
        'UNRATE',     # Unemployment rate - needs update
    ]
    
    # Update all economic indicators
    econ_step = EconomicDataStep()
    context = PipelineContext(data={
        "mode": "backfill",
        "start_date": datetime.now().date() - timedelta(days=365),  # Get last year
        "end_date": datetime.now().date(),
        "symbols": stale_indicators
    })
    
    try:
        result = await econ_step.execute(context)
        records = result.get("total_records", 0)
        logger.info(f"Updated economic data: {records} new records")
        
        # Log which indicators were updated
        if "indicators_updated" in result:
            for indicator, count in result["indicators_updated"].items():
                logger.info(f"  {indicator}: {count} records")
        
        return records
        
    except Exception as e:
        logger.error(f"Failed to update economic data: {e}")
        return 0


async def main():
    """Main function to update all stale data"""
    logger.info("Starting stale data update...")
    logger.info(f"Current time: {datetime.now()}")
    
    # Update market data
    market_records = await update_stale_market_data()
    
    # Update economic data
    econ_records = await update_stale_economic_data()
    
    logger.info("\n=== Update Summary ===")
    logger.info(f"Market data records: {market_records}")
    logger.info(f"Economic data records: {econ_records}")
    logger.info(f"Total records updated: {market_records + econ_records}")
    
    # Run quick check to verify
    logger.info("\nRunning data coverage check...")
    import subprocess
    subprocess.run([sys.executable, "scripts/quick_data_check.py"])


if __name__ == "__main__":
    try:
        asyncio.run(main())
        sys.exit(0)
    except Exception as e:
        logger.error(f"Stale data update failed: {e}")
        sys.exit(1)