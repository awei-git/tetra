"""Pytest configuration and fixtures"""

import pytest
import asyncio
import sys
from pathlib import Path
import os

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_ohlcv_data():
    """Provide sample OHLCV data for testing"""
    from datetime import datetime, timezone
    from decimal import Decimal
    from src.models.market_data import OHLCVData
    
    return OHLCVData(
        symbol="TEST",
        timestamp=datetime.now(timezone.utc),
        open=Decimal("100.00"),
        high=Decimal("105.00"),
        low=Decimal("99.00"),
        close=Decimal("103.00"),
        volume=1000000,
        vwap=Decimal("102.50"),
        trades_count=5000,
        timeframe="1d",
        source="test"
    )


@pytest.fixture
def sample_ohlcv_df():
    """Provide sample OHLCV DataFrame for signal testing"""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
    
    # Generate realistic price movements
    returns = np.random.normal(0.0005, 0.02, 252)
    close_prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'date': dates,
        'close': close_prices
    })
    
    # Generate OHLV from close
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0]) * (1 + np.random.uniform(-0.005, 0.005, 252))
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, 252))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, 252))
    data['volume'] = np.random.randint(1000000, 10000000, 252)
    
    return data.set_index('date')


@pytest.fixture
def multi_symbol_df():
    """Create multi-symbol OHLCV DataFrame"""
    import pandas as pd
    import numpy as np
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    all_data = []
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(0.0005, 0.02, 100)
        close_prices = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'date': dates,
            'symbol': symbol,
            'close': close_prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0]) * (1 + np.random.uniform(-0.005, 0.005, 100))
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, 100))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, 100))
        
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)


@pytest.fixture
def sample_symbols():
    """Provide sample symbols for testing"""
    return ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]


@pytest.fixture
async def mock_db_session():
    """Provide a mock database session"""
    from unittest.mock import AsyncMock
    
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    
    return session


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if --no-integration flag is passed or DB not available"""
    skip_integration = pytest.mark.skip(reason="integration tests skipped")
    
    # Check if database is available
    try:
        import asyncio
        from src.db.base import get_session
        
        async def check_db():
            try:
                async for session in get_session():
                    from sqlalchemy import text
                    await session.execute(text("SELECT 1"))
                    return True
            except:
                return False
        
        db_available = asyncio.run(check_db())
    except:
        db_available = False
    
    for item in items:
        if "integration" in item.keywords:
            if not db_available:
                item.add_marker(skip_integration)