import pytest
from unittest.mock import AsyncMock, patch
from datetime import date

from src.pipelines.data_pipeline.steps.event_data import EventDataStep
from src.pipelines.base import PipelineContext

@pytest.fixture
def event_data_step():
    """Returns an EventDataStep instance."""
    return EventDataStep()

@pytest.mark.asyncio
@patch.object(EventDataStep, '_update_earnings_calendar', new_callable=AsyncMock)
@patch.object(EventDataStep, '_fetch_earnings_from_yfinance', new_callable=AsyncMock)
@patch.object(EventDataStep, '_fetch_dividends_from_yfinance', new_callable=AsyncMock)
@patch.object(EventDataStep, '_fetch_splits_from_yfinance', new_callable=AsyncMock)
@patch('src.pipelines.data_pipeline.steps.event_data.EventDataClient')
async def test_event_data_step_daily_mode(
    MockEventDataClient,
    mock_fetch_splits,
    mock_fetch_dividends,
    mock_fetch_earnings,
    mock_update_earnings,
    event_data_step
):
    """Tests the EventDataStep in daily mode."""
    mock_update_earnings.return_value = 10
    mock_fetch_earnings.return_value = 5
    mock_fetch_dividends.return_value = 2
    mock_fetch_splits.return_value = 1
    mock_event_client = MockEventDataClient.return_value
    mock_event_client.get_market_holidays = AsyncMock(return_value=['2023-01-01'])

    context = PipelineContext()
    context.data = {
        "mode": "daily",
        "symbols": ["AAPL"],
        "start_date": date.today(),
        "end_date": date.today(),
        "fetch_dividends": True,
        "fetch_splits": True
    }

    result = await event_data_step.execute(context)

    assert result["total_records"] > 0
    assert result["earnings"]["success"] == 15
    assert result["dividends"]["success"] == 2
    assert result["splits"]["success"] == 1
    assert result["holidays"]["success"] > 0