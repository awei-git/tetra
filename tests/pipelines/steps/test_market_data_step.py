
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date, timedelta

from src.pipelines.data_pipeline.steps.market_data import MarketDataStep
from src.pipelines.base import PipelineContext

@pytest.fixture
def market_data_step():
    """Returns a MarketDataStep instance."""
    return MarketDataStep()

@pytest.mark.asyncio
@patch('src.pipelines.data_pipeline.steps.market_data.DataIngester')
async def test_market_data_step_daily_mode(MockDataIngester, market_data_step):
    """Tests the MarketDataStep in daily mode."""
    mock_ingester = MockDataIngester.return_value
    mock_ingester.ingest_ohlcv_batch.side_effect = [
        AsyncMock(return_value={
            "total_records": 100,
            "symbols_processed": 2,
            "errors": 0
        })(),
        AsyncMock(return_value={
            "total_records": 100,
            "symbols_processed": 2,
            "errors": 0
        })()
    ]

    context = PipelineContext()
    context.data = {
        "mode": "daily",
        "symbols": ["AAPL", "GOOG"],
        "start_date": date.today(),
        "end_date": date.today(),
        "days_back": 2
    }

    result = await market_data_step.execute(context)

    assert result["total_records"] == 200
    assert len(result["success"]) == 2
    assert len(result["failed"]) == 0
    assert mock_ingester.ingest_ohlcv_batch.call_count == 2

@pytest.mark.asyncio
@patch('src.pipelines.data_pipeline.steps.market_data.DataIngester')
async def test_market_data_step_backfill_mode(MockDataIngester, market_data_step):
    """Tests the MarketDataStep in backfill mode."""
    mock_ingester = MockDataIngester.return_value
    mock_ingester.ingest_ohlcv_batch = AsyncMock(return_value={
        "total_records": 500,
        "symbols_processed": 2,
        "errors": 0
    })

    context = PipelineContext()
    context.data = {
        "mode": "backfill",
        "symbols": ["AAPL", "GOOG"],
        "start_date": date(2023, 1, 1),
        "end_date": date(2023, 1, 31)
    }

    result = await market_data_step.execute(context)

    assert result["total_records"] == 500
    assert len(result["success"]) == 2
    assert len(result["failed"]) == 0
    mock_ingester.ingest_ohlcv_batch.assert_called_once()

@pytest.mark.asyncio
async def test_market_data_step_no_symbols(market_data_step):
    """Tests the MarketDataStep with no symbols."""
    context = PipelineContext()
    context.data = {
        "mode": "daily",
        "symbols": [],
        "start_date": date.today(),
        "end_date": date.today()
    }

    result = await market_data_step.execute(context)

    assert result["total_records"] == 0
    assert len(result["success"]) == 0
    assert len(result["failed"]) == 0
