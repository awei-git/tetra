import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date

from src.pipelines.data_pipeline.pipeline import DataPipeline
from src.pipelines.base import PipelineContext

@pytest.fixture
def pipeline():
    """Returns a DataPipeline instance."""
    return DataPipeline()

@pytest.mark.asyncio
async def test_pipeline_creation(pipeline):
    """Tests that the DataPipeline can be created."""
    assert pipeline is not None
    assert pipeline.name == "DataPipeline"
    assert pipeline.description == "Ingest market data, economic indicators, events, and news"

@pytest.mark.asyncio
@patch('src.pipelines.data_pipeline.pipeline.PipelineContext')
async def test_pipeline_setup_daily_mode(MockPipelineContext, pipeline):
    """Tests the pipeline setup in daily mode."""
    mock_context = MagicMock()
    mock_context.data = {"mode": "daily"}
    MockPipelineContext.return_value = mock_context

    pipeline._get_universe_symbols = AsyncMock(return_value=["AAPL", "GOOG"])
    context = await pipeline.setup()

    assert context.data["mode"] == "daily"
    assert "start_date" in context.data
    assert "end_date" in context.data
    assert context.data["start_date"] == context.data["end_date"]
    assert len(pipeline.steps) > 0

@pytest.mark.asyncio
@patch('src.pipelines.data_pipeline.pipeline.PipelineContext')
async def test_pipeline_setup_backfill_mode(MockPipelineContext, pipeline):
    """Tests the pipeline setup in backfill mode."""
    mock_context = MagicMock()
    mock_context.data = {
        "mode": "backfill",
        "start_date": date(2023, 1, 1),
        "end_date": date(2023, 1, 31)
    }
    MockPipelineContext.return_value = mock_context

    pipeline._get_universe_symbols = AsyncMock(return_value=["AAPL", "GOOG"])

    context = await pipeline.setup()

    assert context.data["mode"] == "backfill"
    assert context.data["start_date"] == date(2023, 1, 1)
    assert context.data["end_date"] == date(2023, 1, 31)
    assert len(pipeline.steps) > 0