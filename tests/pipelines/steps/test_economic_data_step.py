
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date

from src.pipelines.data_pipeline.steps.economic_data import EconomicDataStep
from src.pipelines.base import PipelineContext
from src.definitions.economic_indicators import EconomicIndicators

@pytest.fixture
def economic_data_step():
    """Returns an EconomicDataStep instance."""
    return EconomicDataStep()

@pytest.mark.asyncio
@patch('src.pipelines.data_pipeline.steps.economic_data.EconomicDataClient')
@patch('src.pipelines.data_pipeline.steps.economic_data.get_session')
async def test_economic_data_step_daily_mode(mock_get_session, MockEconomicDataClient, economic_data_step):
    """Tests the EconomicDataStep in daily mode."""
    mock_client = MockEconomicDataClient.return_value.__aenter__.return_value
    mock_client.get_indicator_data = AsyncMock(return_value=[
        MagicMock(symbol='DFF', date=date.today(), value=5.33, source='FRED')
    ])
    mock_db_session = AsyncMock()
    mock_get_session.return_value = mock_db_session

    context = PipelineContext()
    context.data = {
        "mode": "daily",
        "start_date": date.today(),
        "end_date": date.today()
    }

    result = await economic_data_step.execute(context)

    assert result["total_records"] > 0
    assert 'DFF' in result["success"]
    assert len(result["failed"]) == 0
    mock_client.get_indicator_data.assert_called()

@pytest.mark.asyncio
@patch('src.pipelines.data_pipeline.steps.economic_data.EconomicDataClient')
@patch('src.pipelines.data_pipeline.steps.economic_data.get_session')
async def test_economic_data_step_backfill_mode(mock_get_session, MockEconomicDataClient, economic_data_step):
    """Tests the EconomicDataStep in backfill mode."""
    mock_client = MockEconomicDataClient.return_value.__aenter__.return_value
    mock_client.get_indicator_data = AsyncMock(return_value=[
        MagicMock(symbol='GDPC1', date=date(2023, 1, 1), value=20000.0, source='FRED')
    ])
    mock_db_session = AsyncMock()
    mock_get_session.return_value = mock_db_session

    context = PipelineContext()
    context.data = {
        "mode": "backfill",
        "start_date": date(2023, 1, 1),
        "end_date": date(2023, 1, 31)
    }

    result = await economic_data_step.execute(context)

    assert result["total_records"] > 0
    assert 'GDPC1' in result["success"]
    assert len(result["failed"]) == 0
    mock_client.get_indicator_data.assert_called()

@pytest.mark.asyncio
@patch('src.pipelines.data_pipeline.steps.economic_data.EconomicDataClient')
async def test_economic_data_step_api_error(MockEconomicDataClient, economic_data_step):
    """Tests the EconomicDataStep when the API returns an error."""
    mock_client = MockEconomicDataClient.return_value.__aenter__.return_value
    mock_client.get_indicator_data = AsyncMock(side_effect=Exception("API Error"))

    context = PipelineContext()
    context.data = {
        "mode": "daily",
        "start_date": date.today(),
        "end_date": date.today()
    }

    result = await economic_data_step.execute(context)

    assert result["total_records"] == 0
    assert len(result["success"]) == 0
    assert len(result["failed"]) > 0
