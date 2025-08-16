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
        with patch.object(ingester._provider_instance, "fetch_ohlcv", new_callable=AsyncMock) as mock_fetch_ohlcv:
            mock_fetch_ohlcv.return_value = mock_bars
            
            # Mock the database session
            with patch("src.db.base.get_session") as mock_get_session:
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
    async def test_error_handling(self):
        """Test error handling during ingestion"""
        ingester = DataIngester(provider="yfinance")
        
        # Mock the market data client to raise an exception
        with patch.object(ingester._provider_instance, "fetch_ohlcv", new_callable=AsyncMock) as mock_fetch_ohlcv:
            mock_fetch_ohlcv.side_effect = Exception("Connection error")
            
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