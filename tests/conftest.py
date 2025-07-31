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