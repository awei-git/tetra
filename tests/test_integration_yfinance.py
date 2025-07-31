"""Integration test for yfinance data download"""

import pytest
import asyncio
from datetime import date, timedelta

from src.ingestion.data_ingester import DataIngester
from src.db.base import get_session
from src.db.models import OHLCVModel
from sqlalchemy import select, func


class TestYFinanceIntegration:
    """Integration tests for yfinance data ingestion"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_download_single_symbol(self):
        """Test downloading data for a single symbol"""
        ingester = DataIngester(provider="yfinance")
        
        # Download 7 days of Apple data
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
        
        stats = await ingester.ingest_ohlcv_batch(
            symbols=["AAPL"],
            from_date=start_date,
            to_date=end_date,
            timeframe="1d"
        )
        
        print(f"\nDownload stats: {stats}")
        
        # Should have downloaded some data
        assert stats["total_records"] > 0
        assert stats["symbols_processed"] == 1
        assert stats["errors"] == 0
        
        # Verify data in database
        async for session in get_session():
            query = select(func.count()).select_from(OHLCVModel).where(
                OHLCVModel.symbol == "AAPL",
                OHLCVModel.source == "yfinance"
            )
            result = await session.execute(query)
            count = result.scalar()
            
            print(f"Records in database: {count}")
            assert count > 0
            
            # Get a sample record
            sample_query = select(OHLCVModel).where(
                OHLCVModel.symbol == "AAPL",
                OHLCVModel.source == "yfinance"
            ).limit(1)
            
            result = await session.execute(sample_query)
            sample = result.scalar_one_or_none()
            
            if sample:
                print(f"Sample record: {sample.symbol} @ {sample.timestamp}: O={sample.open} H={sample.high} L={sample.low} C={sample.close} V={sample.volume}")
                assert sample.symbol == "AAPL"
                assert sample.source == "yfinance"
                assert sample.open > 0
                assert sample.high >= sample.low
            
            break
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_download_multiple_symbols(self):
        """Test downloading data for multiple symbols"""
        ingester = DataIngester(provider="yfinance")
        
        # Download data for major indices
        symbols = ["SPY", "QQQ", "IWM"]
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        
        stats = await ingester.ingest_ohlcv_batch(
            symbols=symbols,
            from_date=start_date,
            to_date=end_date,
            timeframe="1d"
        )
        
        print(f"\nMulti-symbol download stats: {stats}")
        
        # Should have processed all symbols
        assert stats["symbols_processed"] == len(symbols)
        assert stats["total_records"] > len(symbols) * 10  # At least 10 days per symbol
        
        # Verify total records
        async for session in get_session():
            query = select(func.count()).select_from(OHLCVModel).where(
                OHLCVModel.symbol.in_(symbols),
                OHLCVModel.source == "yfinance"
            )
            result = await session.execute(query)
            total_count = result.scalar()
            
            print(f"Total records for all symbols: {total_count}")
            assert total_count > 0
            
            # Get unique symbols
            query = select(func.count(func.distinct(OHLCVModel.symbol))).where(
                OHLCVModel.symbol.in_(symbols),
                OHLCVModel.source == "yfinance"
            )
            result = await session.execute(query)
            unique_symbols = result.scalar()
            
            print(f"Unique symbols in database: {unique_symbols}")
            # May not get all symbols if some don't have data
            assert unique_symbols > 0
            break
    
    @pytest.mark.asyncio
    @pytest.mark.integration 
    async def test_download_crypto(self):
        """Test downloading crypto data"""
        ingester = DataIngester(provider="yfinance")
        
        # Download Bitcoin data
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
        
        stats = await ingester.ingest_ohlcv_batch(
            symbols=["BTC-USD"],
            from_date=start_date,
            to_date=end_date,
            timeframe="1d"
        )
        
        print(f"\nCrypto download stats: {stats}")
        
        # Crypto trades 7 days a week but may have less data
        assert stats["total_records"] >= 5  # At least 5 days
        assert stats["symbols_processed"] == 1


# Quick test script
if __name__ == "__main__":
    print("Running yfinance integration tests...")
    print("Make sure the database is running: make db-up")
    
    async def run_test():
        test = TestYFinanceIntegration()
        
        print("\n1. Testing single symbol download...")
        await test.test_download_single_symbol()
        
        print("\n2. Testing multiple symbols download...")
        await test.test_download_multiple_symbols()
        
        print("\n3. Testing crypto download...")
        await test.test_download_crypto()
        
        print("\n✅ All tests passed!")
    
    asyncio.run(run_test())