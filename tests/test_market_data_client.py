"""Test market data clients"""

import pytest
import asyncio
from datetime import date, timedelta
from decimal import Decimal

from src.clients.market_data_client import MarketDataClient, YFinanceProvider
from src.models import OHLCVData


class TestMarketDataClient:
    """Test cases for MarketDataClient"""
    
    def test_provider_selection(self):
        """Test that correct provider is selected"""
        # Test polygon provider
        client = MarketDataClient(provider="polygon")
        assert client.provider_name == "polygon"
        
        # Test yfinance provider
        client = MarketDataClient(provider="yfinance")
        assert client.provider_name == "yfinance"
        
        # Test invalid provider
        with pytest.raises(ValueError) as exc_info:
            MarketDataClient(provider="invalid")
        assert "Unknown provider" in str(exc_info.value)
    
    def test_switch_provider(self):
        """Test switching between providers"""
        client = MarketDataClient(provider="polygon")
        assert client.provider_name == "polygon"
        
        client.switch_provider("yfinance")
        assert client.provider_name == "yfinance"


class TestYFinanceProvider:
    """Test cases for YFinance provider"""
    
    @pytest.mark.asyncio
    async def test_get_daily_bars(self):
        """Test fetching daily bars from yfinance"""
        provider = YFinanceProvider()
        
        # Get data for last 7 days
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
        
        async with provider:
            bars = await provider.get_aggregates(
                symbol="AAPL",
                multiplier=1,
                timespan="day",
                from_date=start_date,
                to_date=end_date
            )
        
        # Should have some data
        assert len(bars) > 0
        assert len(bars) <= 7  # At most 7 days
        
        # Check first bar
        if bars:
            first_bar = bars[0]
            assert isinstance(first_bar, OHLCVData)
            assert first_bar.symbol == "AAPL"
            assert first_bar.timeframe == "1d"
            assert first_bar.source == "yfinance"
            
            # Check data types
            assert isinstance(first_bar.open, Decimal)
            assert isinstance(first_bar.high, Decimal)
            assert isinstance(first_bar.low, Decimal)
            assert isinstance(first_bar.close, Decimal)
            assert isinstance(first_bar.volume, int)
            
            # Check price relationships
            assert first_bar.high >= first_bar.low
            assert first_bar.high >= first_bar.open
            assert first_bar.high >= first_bar.close
            assert first_bar.low <= first_bar.open
            assert first_bar.low <= first_bar.close
    
    @pytest.mark.asyncio
    async def test_get_hourly_bars(self):
        """Test fetching hourly bars from yfinance"""
        provider = YFinanceProvider()
        
        # Get data for last 2 days
        end_date = date.today()
        start_date = end_date - timedelta(days=2)
        
        async with provider:
            bars = await provider.get_aggregates(
                symbol="SPY",
                multiplier=1,
                timespan="hour",
                from_date=start_date,
                to_date=end_date
            )
        
        # Should have some data (market hours only)
        assert len(bars) > 0
        
        if bars:
            first_bar = bars[0]
            assert first_bar.symbol == "SPY"
            assert first_bar.timeframe == "1h"
            assert first_bar.source == "yfinance"
    
    @pytest.mark.asyncio
    async def test_invalid_symbol(self):
        """Test handling of invalid symbol"""
        provider = YFinanceProvider()
        
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
        
        async with provider:
            bars = await provider.get_aggregates(
                symbol="INVALID_SYMBOL_12345",
                multiplier=1,
                timespan="day",
                from_date=start_date,
                to_date=end_date
            )
        
        # Should return empty list for invalid symbol
        assert bars == []
    
    @pytest.mark.asyncio
    async def test_crypto_symbol(self):
        """Test fetching crypto data"""
        provider = YFinanceProvider()
        
        end_date = date.today()
        start_date = end_date - timedelta(days=3)
        
        async with provider:
            bars = await provider.get_aggregates(
                symbol="BTC-USD",
                multiplier=1,
                timespan="day",
                from_date=start_date,
                to_date=end_date
            )
        
        # Crypto trades 24/7, should have data
        assert len(bars) > 0
        
        if bars:
            first_bar = bars[0]
            assert first_bar.symbol == "BTC-USD"
            assert first_bar.volume > 0  # Crypto should have volume
    
    @pytest.mark.asyncio
    async def test_tick_data_not_supported(self):
        """Test that tick data returns None"""
        provider = YFinanceProvider()
        
        async with provider:
            tick = await provider.get_last_trade("AAPL")
        
        assert tick is None  # yfinance doesn't support tick data


class TestMarketDataClientIntegration:
    """Integration tests for MarketDataClient with yfinance"""
    
    @pytest.mark.asyncio
    async def test_get_daily_bars_integration(self):
        """Test getting daily bars through MarketDataClient"""
        client = MarketDataClient(provider="yfinance")
        
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        
        async with client:
            bars = await client.get_daily_bars(
                symbol="MSFT",
                from_date=start_date,
                to_date=end_date
            )
        
        # Should have approximately 20-22 trading days in 30 calendar days
        assert len(bars) > 15
        assert len(bars) < 25
        
        # All bars should be daily
        for bar in bars:
            assert bar.timeframe == "1d"
            assert bar.symbol == "MSFT"
    
    @pytest.mark.asyncio
    async def test_get_intraday_bars_integration(self):
        """Test getting intraday bars through MarketDataClient"""
        client = MarketDataClient(provider="yfinance")
        
        end_date = date.today()
        start_date = end_date - timedelta(days=5)
        
        async with client:
            bars = await client.get_intraday_bars(
                symbol="NVDA",
                interval="1h",
                from_date=start_date,
                to_date=end_date
            )
        
        # Should have some hourly data
        assert len(bars) > 0
        
        # All bars should be hourly
        for bar in bars:
            assert bar.timeframe == "1h"
            assert bar.symbol == "NVDA"