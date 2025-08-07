import asyncio
from datetime import datetime, date, timedelta, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from src.clients.market_data_client import MarketDataClient
from src.db.base import get_session, engine
from src.models.sqlalchemy import OHLCVModel
from src.models import OHLCVData
from src.utils.logging import logger
from config import settings


class DataIngester:
    """Handles data ingestion from various sources"""
    
    def __init__(self, provider: str = "polygon"):
        """Initialize data ingester with specified provider"""
        self.provider = provider
        self.batch_size = settings.batch_size
    
    async def ingest_ohlcv_batch(
        self,
        symbols: List[str],
        from_date: date,
        to_date: date,
        timeframe: str = "1d",
        db_session: Optional[AsyncSession] = None
    ) -> Dict[str, int]:
        """
        Ingest OHLCV data for multiple symbols
        
        Args:
            symbols: List of symbols to ingest
            from_date: Start date
            to_date: End date
            timeframe: Timeframe for bars
            db_session: Optional database session
            
        Returns:
            Dictionary with ingestion statistics
        """
        stats = {
            "total_records": 0,
            "inserted": 0,
            "updated": 0,
            "errors": 0,
            "symbols_processed": 0
        }
        
        try:
            async with MarketDataClient(self.provider) as client:
                for symbol in symbols:
                    try:
                        logger.info(f"Ingesting {timeframe} data for {symbol} from {from_date} to {to_date}")
                        
                        # Fetch data based on timeframe
                        if timeframe == "1d":
                            bars = await client.get_daily_bars(symbol, from_date, to_date)
                        else:
                            bars = await client.get_intraday_bars(symbol, timeframe, from_date, to_date)
                        
                        if bars:
                            stats["total_records"] += len(bars)
                            
                            # Save to database
                            if db_session:
                                await self._save_ohlcv_data(bars, db_session, stats)
                            else:
                                async for session in get_session():
                                    await self._save_ohlcv_data(bars, session, stats)
                                    break
                            
                            stats["symbols_processed"] += 1
                            logger.info(f"Ingested {len(bars)} records for {symbol}")
                        else:
                            logger.warning(f"No data found for {symbol} from {self.provider}, trying fallback provider")
                            
                            # Try fallback provider (YFinance) if primary fails
                            if self.provider != "yfinance":
                                try:
                                    async with MarketDataClient("yfinance") as fallback_client:
                                        logger.info(f"Trying YFinance for {symbol}")
                                        
                                        if timeframe == "1d":
                                            bars = await fallback_client.get_daily_bars(symbol, from_date, to_date)
                                        else:
                                            bars = await fallback_client.get_intraday_bars(symbol, timeframe, from_date, to_date)
                                        
                                        if bars:
                                            stats["total_records"] += len(bars)
                                            
                                            # Save to database
                                            if db_session:
                                                await self._save_ohlcv_data(bars, db_session, stats)
                                            else:
                                                async for session in get_session():
                                                    await self._save_ohlcv_data(bars, session, stats)
                                                    break
                                            
                                            stats["symbols_processed"] += 1
                                            logger.info(f"Ingested {len(bars)} records for {symbol} from YFinance fallback")
                                        else:
                                            logger.warning(f"No data found for {symbol} from any provider")
                                except Exception as e:
                                    logger.error(f"Fallback provider failed for {symbol}: {e}")
                        
                        # Small delay between symbols to avoid rate limits
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"Error ingesting data for {symbol}: {e}")
                        stats["errors"] += 1
        except Exception as e:
            logger.error(f"Error initializing market data client: {e}")
            stats["errors"] += 1
        
        logger.info(f"Ingestion complete: {stats}")
        return stats
    
    async def _save_ohlcv_data(
        self,
        bars: List[OHLCVData],
        session: AsyncSession,
        stats: Dict[str, int]
    ):
        """Save OHLCV data to database with upsert logic"""
        try:
            # Process in batches
            for i in range(0, len(bars), self.batch_size):
                batch = bars[i:i + self.batch_size]
                
                # Prepare data for insert
                values = []
                for bar in batch:
                    values.append({
                        "symbol": bar.symbol,
                        "timestamp": bar.timestamp,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "vwap": bar.vwap,
                        "trades_count": bar.trades_count,
                        "timeframe": bar.timeframe,
                        "source": bar.source,
                        "created_at": datetime.now(timezone.utc),
                    })
                
                # Use PostgreSQL upsert
                stmt = insert(OHLCVModel).values(values)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_ohlcv_symbol_time",
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "volume": stmt.excluded.volume,
                        "vwap": stmt.excluded.vwap,
                        "trades_count": stmt.excluded.trades_count,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
                
                await session.execute(stmt)
                await session.commit()
                
                stats["inserted"] += len(batch)
                
        except Exception as e:
            logger.error(f"Error saving OHLCV data: {e}")
            await session.rollback()
            raise
    
    async def backfill_historical_data(
        self,
        symbols: List[str],
        days_back: int = 365,
        timeframe: str = "1d"
    ) -> Dict[str, int]:
        """
        Backfill historical data for symbols
        
        Args:
            symbols: List of symbols
            days_back: Number of days to backfill
            timeframe: Timeframe for data
            
        Returns:
            Ingestion statistics
        """
        to_date = date.today()
        from_date = to_date - timedelta(days=days_back)
        
        logger.info(f"Starting backfill for {len(symbols)} symbols from {from_date} to {to_date}")
        
        return await self.ingest_ohlcv_batch(
            symbols=symbols,
            from_date=from_date,
            to_date=to_date,
            timeframe=timeframe
        )
    
    async def update_latest_data(
        self,
        symbols: List[str],
        timeframe: str = "1d"
    ) -> Dict[str, int]:
        """
        Update with the latest data for symbols
        
        Args:
            symbols: List of symbols
            timeframe: Timeframe for data
            
        Returns:
            Ingestion statistics
        """
        # Get last 7 days to ensure we don't miss anything
        to_date = date.today()
        from_date = to_date - timedelta(days=7)
        
        logger.info(f"Updating latest data for {len(symbols)} symbols")
        
        return await self.ingest_ohlcv_batch(
            symbols=symbols,
            from_date=from_date,
            to_date=to_date,
            timeframe=timeframe
        )


async def main():
    """Example usage of DataIngester"""
    # Example symbols
    symbols = [
        # ETFs
        "SPY", "QQQ", "IWM", "DIA",
        # Large cap stocks
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        # Crypto (if supported by provider)
        # "BTC-USD", "ETH-USD"
    ]
    
    ingester = DataIngester(provider="polygon")
    
    # Backfill historical data
    stats = await ingester.backfill_historical_data(
        symbols=symbols[:2],  # Start with just 2 symbols to test
        days_back=30,  # Last 30 days
        timeframe="1d"
    )
    
    print(f"Backfill completed: {stats}")


if __name__ == "__main__":
    asyncio.run(main())