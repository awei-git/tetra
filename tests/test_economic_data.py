"""Test economic data functionality"""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from src.models.economic_data import EconomicData, EconomicRelease, EconomicForecast
from src.clients.economic_data_client import EconomicDataClient, FREDProvider
from src.data_definitions.economic_indicators import EconomicIndicators, UpdateFrequency


class TestEconomicDataModels:
    """Test economic data models"""
    
    def test_economic_data_model(self):
        """Test EconomicData model creation and validation"""
        data = EconomicData(
            symbol="DFF",
            date=datetime(2024, 1, 15),
            value=Decimal("5.33"),
            source="FRED"
        )
        
        assert data.symbol == "DFF"
        assert data.date == datetime(2024, 1, 15)
        assert data.value == Decimal("5.33")
        assert data.source == "FRED"
        assert data.is_preliminary is False
        assert data.revision_date is None
    
    def test_economic_release_model(self):
        """Test EconomicRelease model with surprise calculation"""
        release = EconomicRelease(
            symbol="CPIAUCSL",
            release_name="Consumer Price Index",
            release_datetime=datetime(2024, 1, 11, 8, 30),
            period="2023-12",
            actual=Decimal("3.4"),
            forecast=Decimal("3.2"),
            previous=Decimal("3.1"),
            impact_level="high"
        )
        
        assert release.symbol == "CPIAUCSL"
        assert release.surprise_magnitude == pytest.approx(0.0625, rel=1e-4)  # (3.4-3.2)/3.2
        
    def test_economic_forecast_model(self):
        """Test EconomicForecast model"""
        forecast = EconomicForecast(
            symbol="GDPC1",
            target_date=datetime(2024, 3, 31),
            forecast_date=datetime(2024, 1, 15),
            forecast_value=Decimal("2.5"),
            forecast_low=Decimal("2.0"),
            forecast_high=Decimal("3.0"),
            source="Fed"
        )
        
        assert forecast.symbol == "GDPC1"
        assert forecast.target_date == datetime(2024, 3, 31)
        assert forecast.forecast_value == Decimal("2.5")


class TestEconomicIndicators:
    """Test economic indicators definitions"""
    
    def test_get_all_indicators(self):
        """Test getting all indicators"""
        indicators = EconomicIndicators.get_all_indicators()
        
        assert len(indicators) > 50  # Should have many indicators
        assert all(len(ind) == 3 for ind in indicators)  # Each should be (symbol, name, frequency)
        
        # Check specific important indicators exist
        symbols = [ind[0] for ind in indicators]
        assert "DFF" in symbols  # Federal Funds Rate
        assert "CPIAUCSL" in symbols  # CPI
        assert "GDPC1" in symbols  # GDP
        assert "UNRATE" in symbols  # Unemployment Rate
    
    def test_get_indicators_by_frequency(self):
        """Test filtering indicators by update frequency"""
        daily = EconomicIndicators.get_daily_indicators()
        monthly = EconomicIndicators.get_monthly_indicators()
        quarterly = EconomicIndicators.get_quarterly_indicators()
        
        # Check frequencies are correct
        assert all(ind[2] == UpdateFrequency.DAILY for ind in daily)
        assert all(ind[2] == UpdateFrequency.MONTHLY for ind in monthly)
        assert all(ind[2] == UpdateFrequency.QUARTERLY for ind in quarterly)
        
        # Should have indicators in each frequency
        assert len(daily) > 5
        assert len(monthly) > 10
        assert len(quarterly) > 3
    
    def test_get_indicators_by_category(self):
        """Test category organization"""
        categories = EconomicIndicators.get_indicators_by_category()
        
        # Check all expected categories exist
        expected_categories = [
            "interest_rates", "inflation", "economic_growth",
            "labor_market", "housing", "consumer_sentiment",
            "manufacturing_business", "monetary_policy",
            "fiscal_policy", "market_indicators", "commodities", "global"
        ]
        
        for category in expected_categories:
            assert category in categories
            assert len(categories[category]) > 0
    
    def test_get_high_priority_indicators(self):
        """Test high priority indicators selection"""
        high_priority = EconomicIndicators.get_high_priority_indicators()
        
        assert len(high_priority) > 10
        assert len(high_priority) < 20
        
        # Should include key indicators
        symbols = [ind[0] for ind in high_priority]
        assert "DFF" in symbols  # Fed Funds
        assert "DGS10" in symbols  # 10-Year Treasury
        assert "CPIAUCSL" in symbols  # CPI
        assert "UNRATE" in symbols  # Unemployment
    
    def test_get_indicator_info(self):
        """Test getting info for specific indicator"""
        # Test known indicator
        info = EconomicIndicators.get_indicator_info("DFF")
        assert info["symbol"] == "DFF"
        assert info["name"] == "Federal Funds Rate"
        assert info["frequency"] == UpdateFrequency.DAILY
        assert info["category"] == "interest_rates"
        
        # Test unknown indicator
        unknown_info = EconomicIndicators.get_indicator_info("UNKNOWN")
        assert unknown_info["symbol"] == "UNKNOWN"
        assert unknown_info["category"] == "unknown"
        assert unknown_info["frequency"] is None


class TestFREDProvider:
    """Test FRED provider functionality"""
    
    @pytest.mark.asyncio
    async def test_fred_provider_init(self):
        """Test FRED provider initialization"""
        # Mock the settings object attributes
        with patch('src.clients.economic_data_client.settings') as mock_settings:
            # Configure the mock to have the required attributes
            mock_settings.configure_mock(
                fred_api_key='test_key',
                fred_base_url='https://api.stlouisfed.org/fred',
                fred_rate_limit=120
            )
            
            provider = FREDProvider()
            assert provider.api_key == 'test_key'
            assert provider.base_url == "https://api.stlouisfed.org/fred"
    
    @pytest.mark.asyncio
    async def test_fred_provider_no_api_key(self):
        """Test FRED provider without API key"""
        with patch('src.clients.economic_data_client.settings') as mock_settings:
            mock_settings.configure_mock(fred_api_key=None)
            
            with pytest.raises(ValueError, match="FRED API key not provided"):
                FREDProvider()
    
    @pytest.mark.asyncio
    async def test_get_indicator_data(self):
        """Test fetching indicator data from FRED"""
        with patch('src.clients.economic_data_client.settings') as mock_settings:
            mock_settings.configure_mock(
                fred_api_key='test_key',
                fred_base_url='https://api.stlouisfed.org/fred',
                fred_rate_limit=120
            )
            
            provider = FREDProvider()
            
            # Mock the HTTP response
            mock_response = {
                "observations": [
                    {"date": "2024-01-01", "value": "5.33"},
                    {"date": "2024-01-02", "value": "5.33"},
                    {"date": "2024-01-03", "value": "."},  # Missing value
                    {"date": "2024-01-04", "value": "5.35"}
                ]
            }
            
            provider._make_fred_request = AsyncMock(return_value=mock_response)
            
            # Test fetching data
            data = await provider.get_indicator_data(
                "DFF",
                from_date=date(2024, 1, 1),
                to_date=date(2024, 1, 4)
            )
            
            assert len(data) == 3  # Should skip missing value
            assert data[0].symbol == "DFF"
            assert data[0].date == datetime(2024, 1, 1)
            assert data[0].value == Decimal("5.33")
            assert data[0].source == "FRED"
            
            # Verify request was made correctly
            provider._make_fred_request.assert_called_once()
            call_args = provider._make_fred_request.call_args
            assert call_args.args[0] == "series/observations"
            # params is passed as second positional argument
            assert call_args.args[1]["series_id"] == "DFF"


class TestEconomicDataClient:
    """Test main economic data client"""
    
    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization with different providers"""
        with patch('src.clients.economic_data_client.settings') as mock_settings:
            mock_settings.configure_mock(
                fred_api_key='test_key',
                fred_base_url='https://api.stlouisfed.org/fred',
                fred_rate_limit=120
            )
            
            client = EconomicDataClient(provider="fred")
            assert client.provider_name == "fred"
            assert isinstance(client.provider, FREDProvider)
    
    @pytest.mark.asyncio
    async def test_client_unknown_provider(self):
        """Test client with unknown provider"""
        with pytest.raises(ValueError, match="Unknown provider: unknown"):
            EconomicDataClient(provider="unknown")
    
    @pytest.mark.asyncio
    async def test_get_multiple_indicators(self):
        """Test fetching multiple indicators"""
        with patch('src.clients.economic_data_client.settings') as mock_settings:
            mock_settings.configure_mock(
                fred_api_key='test_key',
                fred_base_url='https://api.stlouisfed.org/fred',
                fred_rate_limit=120
            )
            
            client = EconomicDataClient(provider="fred")
            
            # Mock the provider
            mock_data_dff = [
                EconomicData(symbol="DFF", date=datetime(2024, 1, 1), value=Decimal("5.33"), source="FRED")
            ]
            mock_data_dgs10 = [
                EconomicData(symbol="DGS10", date=datetime(2024, 1, 1), value=Decimal("4.02"), source="FRED")
            ]
            
            client.provider.get_indicator_data = AsyncMock(
                side_effect=[mock_data_dff, mock_data_dgs10]
            )
            
            # Test fetching multiple
            data = await client.get_multiple_indicators(
                ["DFF", "DGS10"],
                from_date=date(2024, 1, 1),
                to_date=date(2024, 1, 31)
            )
            
            assert len(data) == 2
            assert "DFF" in data
            assert "DGS10" in data
            assert len(data["DFF"]) == 1
            assert len(data["DGS10"]) == 1
            assert data["DFF"][0].value == Decimal("5.33")
            assert data["DGS10"][0].value == Decimal("4.02")
    
    @pytest.mark.asyncio
    async def test_get_multiple_indicators_with_error(self):
        """Test handling errors when fetching multiple indicators"""
        with patch('src.clients.economic_data_client.settings') as mock_settings:
            mock_settings.configure_mock(
                fred_api_key='test_key',
                fred_base_url='https://api.stlouisfed.org/fred',
                fred_rate_limit=120
            )
            
            client = EconomicDataClient(provider="fred")
            
            # Mock one success and one failure
            mock_data = [
                EconomicData(symbol="DFF", date=datetime(2024, 1, 1), value=Decimal("5.33"), source="FRED")
            ]
            
            client.provider.get_indicator_data = AsyncMock(
                side_effect=[mock_data, Exception("API Error")]
            )
            
            # Test fetching with error
            data = await client.get_multiple_indicators(
                ["DFF", "INVALID"],
                from_date=date(2024, 1, 1)
            )
            
            assert len(data) == 2
            assert len(data["DFF"]) == 1
            assert data["INVALID"] == []  # Error results in empty list


class TestIntegration:
    """Integration tests for economic data flow"""
    
    @pytest.mark.asyncio
    async def test_full_data_flow(self):
        """Test complete flow from fetching to models"""
        with patch('src.clients.economic_data_client.settings') as mock_settings:
            mock_settings.configure_mock(
                fred_api_key='test_key',
                fred_base_url='https://api.stlouisfed.org/fred',
                fred_rate_limit=120
            )
            
            client = EconomicDataClient(provider="fred")
            
            # Mock FRED response
            mock_response = {
                "observations": [
                    {"date": "2024-01-01", "value": "5.33"},
                    {"date": "2024-01-02", "value": "5.33"},
                ]
            }
            
            client.provider._make_fred_request = AsyncMock(return_value=mock_response)
            
            # Get data
            data = await client.get_indicator_data(
                "DFF",
                from_date=date(2024, 1, 1),
                to_date=date(2024, 1, 2)
            )
            
            # Verify data
            assert len(data) == 2
            assert all(isinstance(item, EconomicData) for item in data)
            assert data[0].symbol == "DFF"
            assert data[0].value == Decimal("5.33")
            
            # Verify data can be serialized
            json_data = data[0].model_dump_json()
            assert "5.33" in json_data
            assert "2024-01-01" in json_data