"""Test the universe module"""

import pytest
from src.definitions.market_universe import MarketUniverse as Universe


class TestUniverse:
    """Test cases for the Universe class"""
    
    def test_get_all_etfs(self):
        """Test that ETF lists are properly combined"""
        etfs = Universe.get_all_etfs()
        
        # Should have unique symbols
        assert len(etfs) == len(set(etfs))
        
        # Should contain known ETFs
        assert "SPY" in etfs
        assert "QQQ" in etfs
        assert "ARKK" in etfs
        
        # Should have reasonable number of ETFs
        assert len(etfs) > 50
        assert len(etfs) < 200
    
    def test_get_all_stocks(self):
        """Test that stock lists are properly combined"""
        stocks = Universe.get_all_stocks()
        
        # Should have unique symbols
        assert len(stocks) == len(set(stocks))
        
        # Should contain known stocks
        assert "AAPL" in stocks
        assert "MSFT" in stocks
        assert "NVDA" in stocks
        
        # Should have reasonable number of stocks
        assert len(stocks) > 50
        assert len(stocks) < 300
    
    def test_get_all_symbols(self):
        """Test that all symbols are properly combined"""
        all_symbols = Universe.get_all_symbols()
        etfs = Universe.get_all_etfs()
        stocks = Universe.get_all_stocks()
        crypto = Universe.CRYPTO_SYMBOLS
        
        # Total should equal sum of categories
        assert len(all_symbols) == len(etfs) + len(stocks) + len(crypto)
        
        # Should contain symbols from each category
        assert "SPY" in all_symbols  # ETF
        assert "AAPL" in all_symbols  # Stock
        assert "BTC-USD" in all_symbols  # Crypto
    
    def test_high_priority_symbols(self):
        """Test high priority symbols selection"""
        high_priority = Universe.get_high_priority_symbols()
        
        # Should have reasonable number
        assert len(high_priority) > 10
        assert len(high_priority) < 30
        
        # Should include major indices and top stocks
        assert "SPY" in high_priority
        assert "QQQ" in high_priority
        assert "AAPL" in high_priority
        assert "MSFT" in high_priority
    
    def test_is_crypto(self):
        """Test crypto symbol identification"""
        assert Universe.is_crypto("BTC-USD") is True
        assert Universe.is_crypto("ETH-USD") is True
        assert Universe.is_crypto("AAPL") is False
        assert Universe.is_crypto("SPY") is False
    
    def test_is_etf(self):
        """Test ETF symbol identification"""
        assert Universe.is_etf("SPY") is True
        assert Universe.is_etf("QQQ") is True
        assert Universe.is_etf("ARKK") is True
        assert Universe.is_etf("AAPL") is False
        assert Universe.is_etf("BTC-USD") is False
    
    def test_get_symbol_info(self):
        """Test symbol information retrieval"""
        # Test ETF
        spy_info = Universe.get_symbol_info("SPY")
        assert spy_info["symbol"] == "SPY"
        assert spy_info["is_etf"] is True
        assert spy_info["is_crypto"] is False
        assert spy_info["category"] == "index_etfs"
        
        # Test Stock
        aapl_info = Universe.get_symbol_info("AAPL")
        assert aapl_info["symbol"] == "AAPL"
        assert aapl_info["is_etf"] is False
        assert aapl_info["is_crypto"] is False
        assert aapl_info["category"] == "large_cap_stocks"
        
        # Test Crypto
        btc_info = Universe.get_symbol_info("BTC-USD")
        assert btc_info["symbol"] == "BTC-USD"
        assert btc_info["is_etf"] is False
        assert btc_info["is_crypto"] is True
        assert btc_info["category"] == "crypto"
        
        # Test unknown symbol
        unknown_info = Universe.get_symbol_info("UNKNOWN")
        assert unknown_info["symbol"] == "UNKNOWN"
        assert unknown_info["is_etf"] is False
        assert unknown_info["is_crypto"] is False
        assert unknown_info["category"] == "unknown"
    
    def test_universe_by_category(self):
        """Test category organization"""
        categories = Universe.get_universe_by_category()
        
        # Should have all expected categories
        expected_categories = [
            "index_etfs", "sector_etfs", "international_etfs",
            "bond_etfs", "commodity_etfs", "thematic_etfs",
            "reit_etfs", "volatility_etfs", "large_cap_stocks", "quantum_computing",
            "defense", "ai_infrastructure", "consumer",
            "crypto_stocks", "crypto"
        ]
        
        for category in expected_categories:
            assert category in categories
            assert len(categories[category]) > 0
    
    def test_no_duplicates_within_categories(self):
        """Test that there are no duplicates within each category"""
        # Check each category list for duplicates
        assert len(Universe.INDEX_ETFS) == len(set(Universe.INDEX_ETFS))
        assert len(Universe.SECTOR_ETFS) == len(set(Universe.SECTOR_ETFS))
        assert len(Universe.LARGE_CAP_STOCKS) == len(set(Universe.LARGE_CAP_STOCKS))
        assert len(Universe.CRYPTO_SYMBOLS) == len(set(Universe.CRYPTO_SYMBOLS))