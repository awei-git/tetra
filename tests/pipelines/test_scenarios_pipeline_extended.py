import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import date, timedelta
import pandas as pd

from src.pipelines.scenarios_pipeline.pipeline import ScenariosPipeline
from src.pipelines.base import PipelineContext

@pytest.fixture
def scenarios_pipeline():
    """Returns a ScenariosPipeline instance for testing."""
    config = {
        'scenario_types': ['historical'],
        'storage': {
            'save_to_database': False,
            'save_timeseries': False,
            'save_metadata': True
        }
    }
    return ScenariosPipeline(config)

@pytest.mark.asyncio
@patch('src.db.base.async_session_maker')
@patch('src.pipelines.scenarios_pipeline.pipeline.ScenarioStorageStep') # Patch the ScenarioStorageStep
async def test_scenarios_pipeline_historical_generation(
    MockScenarioStorageStep,
    mock_async_session_maker,
    scenarios_pipeline
):
    """Tests the ScenariosPipeline's ability to generate historical scenarios."""
    # Mock get_session for database interactions (e.g., symbol fetching)
    mock_session_instance = AsyncMock()
    mock_session_instance.execute.side_effect = [
        MagicMock(fetchall=lambda: [MagicMock(symbol='AAPL')]), # For SELECT DISTINCT symbol
        MagicMock(fetchall=lambda: [ # For OHLCV data
            MagicMock(symbol='AAPL', date=date(2020, 1, 1), open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0, vwap=100.0),
            MagicMock(symbol='AAPL', date=date(2020, 1, 2), open=100.0, high=102.0, low=100.0, close=101.0, volume=1000.0, vwap=101.0),
            MagicMock(symbol='AAPL', date=date(2020, 1, 3), open=101.0, high=103.0, low=101.0, close=102.0, volume=1000.0, vwap=102.0),
            MagicMock(symbol='AAPL', date=date(2020, 1, 4), open=102.0, high=104.0, low=102.0, close=103.0, volume=1000.0, vwap=103.0),
            MagicMock(symbol='AAPL', date=date(2020, 1, 5), open=103.0, high=105.0, low=103.0, close=104.0, volume=1000.0, vwap=104.0),
            MagicMock(symbol='AAPL', date=date(2020, 1, 6), open=104.0, high=106.0, low=104.0, close=105.0, volume=1000.0, vwap=105.0),
            MagicMock(symbol='AAPL', date=date(2020, 1, 7), open=105.0, high=107.0, low=105.0, close=106.0, volume=1000.0, vwap=106.0),
            MagicMock(symbol='AAPL', date=date(2020, 1, 8), open=106.0, high=108.0, low=106.0, close=107.0, volume=1000.0, vwap=107.0),
            MagicMock(symbol='AAPL', date=date(2020, 1, 9), open=107.0, high=109.0, low=107.0, close=108.0, volume=1000.0, vwap=108.0),
            MagicMock(symbol='AAPL', date=date(2020, 1, 10), open=108.0, high=110.0, low=108.0, close=109.0, volume=1000.0, vwap=109.0)
        ])
    ]
    mock_async_session_maker.return_value.__aenter__.return_value = mock_session_instance
    mock_async_session_maker.return_value.__aexit__.return_value = None
    mock_async_session_maker.return_value.__aexit__.return_value = None
    mock_async_session_maker.return_value.__aexit__.return_value = None

    # Mock ScenarioStorageStep's execute method
    mock_storage_step_instance = MockScenarioStorageStep.return_value
    mock_storage_step_instance.execute = AsyncMock(return_value=None) # Prevent actual storage

    # Run the pipeline
    context = PipelineContext()
    context.data["start_date"] = date(2020, 1, 1)
    context.data["end_date"] = date(2020, 1, 10)
    context.data["symbols"] = ["AAPL"]

    context = await scenarios_pipeline.run(context=context)

    # Assertions
    assert mock_session_instance.execute.called
    assert mock_storage_step_instance.execute.called # Verify storage step was called
    assert len(context.data["scenarios"]) > 0
    assert any(s.scenario_type == 'crisis' for s in context.data["scenarios"])