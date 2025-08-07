"""Test data ingestion functionality"""

import pytest
import asyncio
from datetime import date, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.ingestion.data_ingester import DataIngester
from src.models import OHLCVData
from decimal import Decimal


class TestDataIngester:
    """Test cases for DataIngester"""
    
    def test_initialization(self):
        """Test DataIngester initialization"""
        # Test with default provider
        ingester = DataIngester()
        assert ingester.provider == "polygon"
        assert ingester.batch_size > 0
        
        # Test with yfinance provider
        ingester = DataIngester(provider="yfinance")
        assert ingester.provider == "yfinance"
    
    @pytest.mark.asyncio
    async def test_ingest_ohlcv_batch_mock(self):
        """Test ingesting OHLCV data with mocked client"""
        ingester = DataIngester(provider="yfinance")
        
        # Mock data
        mock_bars = [
            OHLCVData(
                symbol="TEST",
                timestamp=datetime.now(),
                open=Decimal("100"),
                high=Decimal("105"),
                low=Decimal("99"),
                close=Decimal("103"),
                volume=1000000,
                timeframe="1d",
                source="yfinance"
            )
        ]
        
        # Mock the market data client
        with patch("src.ingestion.data_ingester.MarketDataClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get_daily_bars.return_value = mock_bars
            mock_client_class.return_value = mock_client
            
            # Mock the database session
            with patch("src.ingestion.data_ingester.get_session") as mock_get_session:
                mock_session = AsyncMock()
                mock_get_session.return_value.__aiter__.return_value = [mock_session]
                
                # Run ingestion
                stats = await ingester.ingest_ohlcv_batch(
                    symbols=["TEST"],
                    from_date=date.today() - timedelta(days=7),
                    to_date=date.today(),
                    timeframe="1d"
                )
                
                # Check results
                assert stats["total_records"] == 1
                assert stats["symbols_processed"] == 1
                assert stats["errors"] == 0
    
    @pytest.mark.asyncio
    async def test_backfill_historical_data(self):
        """Test backfill functionality"""
        ingester = DataIngester(provider="yfinance")
        
        with patch.object(ingester, "ingest_ohlcv_batch") as mock_ingest:
            mock_ingest.return_value = {
                "total_records": 365,
                "inserted": 365,
                "updated": 0,
                "errors": 0,
                "symbols_processed": 1
            }
            
            stats = await ingester.backfill_historical_data(
                symbols=["AAPL"],
                days_back=365,
                timeframe="1d"
            )
            
            # Check that ingest was called with correct parameters
            mock_ingest.assert_called_once()
            call_args = mock_ingest.call_args[1]
            assert call_args["symbols"] == ["AAPL"]
            assert call_args["timeframe"] == "1d"
            assert (date.today() - call_args["from_date"]).days == 365
            assert call_args["to_date"] == date.today()
    
    @pytest.mark.asyncio
    async def test_update_latest_data(self):
        """Test updating latest data"""
        ingester = DataIngester(provider="yfinance")
        
        with patch.object(ingester, "ingest_ohlcv_batch") as mock_ingest:
            mock_ingest.return_value = {
                "total_records": 7,
                "inserted": 5,
                "updated": 2,
                "errors": 0,
                "symbols_processed": 1
            }
            
            stats = await ingester.update_latest_data(
                symbols=["SPY"],
                timeframe="1d"
            )
            
            # Check that ingest was called with last 7 days
            mock_ingest.assert_called_once()
            call_args = mock_ingest.call_args[1]
            assert call_args["symbols"] == ["SPY"]
            assert call_args["timeframe"] == "1d"
            assert (date.today() - call_args["from_date"]).days == 7
            assert call_args["to_date"] == date.today()
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling during ingestion"""
        ingester = DataIngester(provider="yfinance")
        
        # Mock the market data client to raise an exception
        with patch("src.ingestion.data_ingester.MarketDataClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.side_effect = Exception("Connection error")
            mock_client_class.return_value = mock_client
            
            stats = await ingester.ingest_ohlcv_batch(
                symbols=["ERROR"],
                from_date=date.today() - timedelta(days=1),
                to_date=date.today(),
                timeframe="1d"
            )
            
            # Should handle error gracefully
            assert stats["total_records"] == 0
            assert stats["errors"] == 1
            assert stats["symbols_processed"] == 0


# Import datetime for the mock data
from datetime import datetime