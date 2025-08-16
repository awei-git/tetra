
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date

from src.pipelines.data_pipeline.steps.news_sentiment import NewsSentimentStep
from src.pipelines.base import PipelineContext

@pytest.fixture
def news_sentiment_step():
    """Returns a NewsSentimentStep instance."""
    return NewsSentimentStep()

@pytest.mark.asyncio
@patch('src.pipelines.data_pipeline.steps.news_sentiment.NewsSentimentClient')
@patch('src.pipelines.data_pipeline.steps.news_sentiment.get_session')
async def test_news_sentiment_step_newsapi_provider(mock_get_session, MockNewsSentimentClient, news_sentiment_step):
    """Tests the NewsSentimentStep with the newsapi provider."""
    mock_client = MockNewsSentimentClient.return_value.__aenter__.return_value
    mock_client.get_news = AsyncMock(return_value=[
        MagicMock(url='http://test.com/1', title='Test Article 1', author='Test Author 1', source='Test Source 1', published_at=date.today(), content='Test Content 1', symbols=['AAPL']),
        MagicMock(url='http://test.com/2', title='Test Article 2', author='Test Author 2', source='Test Source 2', published_at=date.today(), content='Test Content 2', symbols=['GOOG'])
    ])
    mock_db_session = AsyncMock()
    mock_get_session.return_value = mock_db_session

    context = PipelineContext()
    context.data = {
        "mode": "daily",
        "symbols": ["AAPL", "GOOG"],
        "start_date": date.today(),
        "end_date": date.today(),
        "news_provider": "newsapi"
    }

    result = await news_sentiment_step.execute(context)

    assert result["articles"] == 2
    assert result["symbols_processed"] == 2
    assert len(result["errors"]) == 0
    mock_client.get_news.assert_called()

@pytest.mark.asyncio
@patch('src.pipelines.data_pipeline.steps.news_sentiment.NewsSentimentClient')
async def test_news_sentiment_step_alphavantage_provider(MockNewsSentimentClient, news_sentiment_step):
    """Tests the NewsSentimentStep with the alphavantage provider."""
    mock_client = MockNewsSentimentClient.return_value.__aenter__.return_value
    mock_client.get_sentiment = AsyncMock(return_value=[
        MagicMock(sentiment=0.5, source='Test Source'),
        MagicMock(sentiment=-0.5, source='Test Source')
    ])

    context = PipelineContext()
    context.data = {
        "mode": "daily",
        "symbols": ["AAPL", "GOOG"],
        "start_date": date.today(),
        "end_date": date.today(),
        "news_provider": "alphavantage"
    }

    result = await news_sentiment_step.execute(context)

    assert result["sentiments"] > 0
    assert result["symbols_processed"] > 0
    assert len(result["errors"]) == 0
    mock_client.get_sentiment.assert_called()
