"""Tests for news sentiment functionality"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from decimal import Decimal

from src.clients.news_sentiment_client import (
    NewsSentimentClient, NewsAPIProvider, AlphaVantageNewsProvider
)
from src.models import (
    NewsArticle, SentimentScore, NewsSentiment, NewsSource,
    SentimentType, NewsCategory
)
from src.models.sqlalchemy.news_sentiment import NewsArticleModel, NewsSentimentModel
from src.utils.logging import logger


@pytest.fixture
def mock_settings():
    """Mock settings"""
    with patch('src.clients.news_sentiment_client.settings') as mock:
        mock.news_api_key = 'test_news_api_key'
        mock.alphavantage_api_key = 'test_av_key'
        yield mock


@pytest.fixture
def mock_newsapi_response():
    """Sample NewsAPI response"""
    return {
        "status": "ok",
        "totalResults": 2,
        "articles": [
            {
                "source": {"id": "bloomberg", "name": "Bloomberg"},
                "author": "John Doe",
                "title": "Apple Reports Strong Q4 Earnings",
                "description": "Apple Inc. reported better than expected earnings...",
                "url": "https://bloomberg.com/apple-earnings",
                "urlToImage": "https://bloomberg.com/image.jpg",
                "publishedAt": "2025-07-30T10:00:00Z",
                "content": "Full article content here..."
            },
            {
                "source": {"id": "reuters", "name": "Reuters"},
                "author": "Jane Smith",
                "title": "Tesla Announces New Factory",
                "description": "Tesla plans to build new factory...",
                "url": "https://reuters.com/tesla-factory",
                "urlToImage": None,
                "publishedAt": "2025-07-29T14:30:00Z",
                "content": "Tesla content here..."
            }
        ]
    }


@pytest.fixture
def mock_alphavantage_response():
    """Sample Alpha Vantage response"""
    return {
        "items": "50",
        "sentiment_score_definition": "x <= -0.35: Bearish; -0.35 < x <= -0.15: Somewhat-Bearish; -0.15 < x < 0.15: Neutral; 0.15 <= x < 0.35: Somewhat_Bullish; x >= 0.35: Bullish",
        "relevance_score_definition": "0 < x <= 1, with a higher score indicating higher relevance.",
        "feed": [
            {
                "title": "Apple Beats Earnings Expectations",
                "url": "https://example.com/apple-earnings",
                "time_published": "20250730T100000",
                "authors": [],
                "summary": "Apple reported strong earnings...",
                "banner_image": "https://example.com/apple.jpg",
                "source": "MarketWatch",
                "category_within_source": "n/a",
                "source_domain": "marketwatch.com",
                "topics": [
                    {"topic": "Earnings", "relevance_score": "0.95"},
                    {"topic": "Technology", "relevance_score": "0.80"}
                ],
                "overall_sentiment_score": 0.45,
                "overall_sentiment_label": "Bullish",
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": "0.95",
                        "ticker_sentiment_score": "0.50",
                        "ticker_sentiment_label": "Bullish"
                    }
                ]
            },
            {
                "title": "Market Concerns Over Tech Valuations",
                "url": "https://example.com/tech-valuations",
                "time_published": "20250729T143000",
                "authors": [{"name": "John Analyst"}],
                "summary": "Investors worried about tech valuations...",
                "banner_image": None,
                "source": "CNBC",
                "category_within_source": "n/a",
                "source_domain": "cnbc.com",
                "topics": [
                    {"topic": "Financial_Markets", "relevance_score": "0.90"}
                ],
                "overall_sentiment_score": -0.25,
                "overall_sentiment_label": "Somewhat-Bearish",
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": "0.60",
                        "ticker_sentiment_score": "-0.20",
                        "ticker_sentiment_label": "Somewhat-Bearish"
                    }
                ]
            }
        ]
    }


class TestNewsAPIProvider:
    """Tests for NewsAPI provider"""
    
    @pytest.mark.asyncio
    async def test_init(self, mock_settings):
        """Test NewsAPIProvider initialization"""
        provider = NewsAPIProvider()
        assert provider.api_key == 'test_news_api_key'
        assert provider.base_url == "https://newsapi.org/v2"
        assert provider.rate_limiter.calls == 500
        assert provider.rate_limiter.period == 86400
    
    @pytest.mark.asyncio
    async def test_init_without_key(self, mock_settings):
        """Test initialization without API key"""
        mock_settings.news_api_key = None
        with pytest.raises(ValueError, match="NewsAPI key not provided"):
            NewsAPIProvider()
    
    @pytest.mark.asyncio
    async def test_get_news_success(self, mock_settings, mock_newsapi_response):
        """Test successful news fetching"""
        provider = NewsAPIProvider()
        
        # Mock the get method
        provider.get = AsyncMock(return_value=mock_newsapi_response)
        
        # Fetch news
        articles = await provider.get_news(
            symbols=["AAPL", "TSLA"],
            from_date=date(2025, 7, 29),
            to_date=date(2025, 7, 30)
        )
        
        # Verify results
        assert len(articles) == 2
        
        # Check first article
        article1 = articles[0]
        assert article1.source == "Bloomberg"
        assert article1.title == "Apple Reports Strong Q4 Earnings"
        assert article1.author == "John Doe"
        assert article1.source_category == NewsSource.NEWSAPI
        
        # Check second article
        article2 = articles[1]
        assert article2.source == "Reuters"
        assert article2.title == "Tesla Announces New Factory"
    
    @pytest.mark.asyncio
    async def test_get_news_error_response(self, mock_settings):
        """Test handling of error response"""
        provider = NewsAPIProvider()
        
        # Mock error response
        provider.get = AsyncMock(return_value={"status": "error", "message": "Invalid API key"})
        
        articles = await provider.get_news(symbols=["AAPL"])
        assert articles == []
    
    @pytest.mark.asyncio
    async def test_get_news_exception(self, mock_settings):
        """Test exception handling"""
        provider = NewsAPIProvider()
        
        # Mock exception
        provider.get = AsyncMock(side_effect=Exception("Network error"))
        
        articles = await provider.get_news(symbols=["AAPL"])
        assert articles == []
    
    @pytest.mark.asyncio
    async def test_get_sentiment_not_supported(self, mock_settings):
        """Test that sentiment is not supported"""
        provider = NewsAPIProvider()
        
        sentiments = await provider.get_sentiment(symbols=["AAPL"])
        assert sentiments == []


class TestAlphaVantageNewsProvider:
    """Tests for Alpha Vantage provider"""
    
    @pytest.mark.asyncio
    async def test_init(self, mock_settings):
        """Test AlphaVantageNewsProvider initialization"""
        provider = AlphaVantageNewsProvider()
        assert provider.api_key == 'test_av_key'
        assert provider.base_url == "https://www.alphavantage.co"
        assert provider.rate_limiter.calls == 5
        assert provider.rate_limiter.period == 60
    
    @pytest.mark.asyncio
    async def test_get_sentiment_success(self, mock_settings, mock_alphavantage_response):
        """Test successful sentiment fetching"""
        provider = AlphaVantageNewsProvider()
        
        # Mock the get method
        provider.get = AsyncMock(return_value=mock_alphavantage_response)
        
        # Fetch sentiments
        sentiments = await provider.get_sentiment(
            symbols=["AAPL"],
            from_date=date(2025, 7, 29),
            to_date=date(2025, 7, 30)
        )
        
        # Verify results
        assert len(sentiments) == 2
        
        # Check first sentiment
        sentiment1 = sentiments[0]
        assert sentiment1.article.title == "Apple Beats Earnings Expectations"
        assert sentiment1.article.source == "MarketWatch"
        assert sentiment1.overall_sentiment.polarity == 0.5  # Ticker sentiment score
        assert sentiment1.overall_sentiment.positive > 0
        assert sentiment1.overall_sentiment.bullish == 0.5
        assert sentiment1.relevance_score == 0.95
        assert sentiment1.sentiment_model == "alphavantage"
        
        # Check second sentiment
        sentiment2 = sentiments[1]
        assert sentiment2.article.title == "Market Concerns Over Tech Valuations"
        assert sentiment2.overall_sentiment.polarity == -0.2  # Ticker sentiment score
        assert sentiment2.overall_sentiment.negative > 0
        assert sentiment2.relevance_score == 0.6
    
    @pytest.mark.asyncio
    async def test_get_sentiment_no_symbols(self, mock_settings):
        """Test sentiment request without symbols"""
        provider = AlphaVantageNewsProvider()
        
        sentiments = await provider.get_sentiment(symbols=[])
        assert sentiments == []
    
    @pytest.mark.asyncio
    async def test_get_sentiment_error(self, mock_settings):
        """Test error handling"""
        provider = AlphaVantageNewsProvider()
        
        # Mock error response
        provider.get = AsyncMock(return_value={"Error Message": "Invalid API key"})
        
        sentiments = await provider.get_sentiment(symbols=["AAPL"])
        assert sentiments == []
    
    @pytest.mark.asyncio
    async def test_get_news_delegates_to_sentiment(self, mock_settings, mock_alphavantage_response):
        """Test that get_news delegates to get_sentiment"""
        provider = AlphaVantageNewsProvider()
        
        # Mock the get method
        provider.get = AsyncMock(return_value=mock_alphavantage_response)
        
        # Fetch news
        articles = await provider.get_news(symbols=["AAPL"])
        
        # Should return articles extracted from sentiments
        assert len(articles) == 2
        assert articles[0].title == "Apple Beats Earnings Expectations"
        assert articles[1].title == "Market Concerns Over Tech Valuations"
    
    @pytest.mark.asyncio
    async def test_map_topics_to_categories(self, mock_settings):
        """Test topic to category mapping"""
        provider = AlphaVantageNewsProvider()
        
        topics = [
            {"topic": "Earnings"},
            {"topic": "Mergers_and_Acquisitions"},
            {"topic": "Financial_Markets"},
            {"topic": "Economy_Macro"},
            {"topic": "Technology"}
        ]
        
        categories = provider._map_topics_to_categories(topics)
        
        assert NewsCategory.EARNINGS in categories
        assert NewsCategory.MERGER in categories
        assert NewsCategory.MARKET in categories
        assert NewsCategory.MACRO in categories
        assert NewsCategory.OTHER in categories
    
    @pytest.mark.asyncio
    async def test_estimate_impact(self, mock_settings):
        """Test impact score estimation"""
        provider = AlphaVantageNewsProvider()
        
        assert provider._estimate_impact(0.6, "Bullish") == 0.8
        assert provider._estimate_impact(0.4, "Somewhat-Bullish") == 0.6
        assert provider._estimate_impact(-0.2, "Somewhat-Bearish") == 0.4
        assert provider._estimate_impact(0.05, "Neutral") == 0.2


class TestNewsSentimentClient:
    """Tests for the main news sentiment client"""
    
    @pytest.mark.asyncio
    async def test_init_alphavantage(self, mock_settings):
        """Test initialization with AlphaVantage provider"""
        client = NewsSentimentClient(provider="alphavantage")
        assert client.provider_name == "alphavantage"
        assert isinstance(client.provider, AlphaVantageNewsProvider)
    
    @pytest.mark.asyncio
    async def test_init_newsapi(self, mock_settings):
        """Test initialization with NewsAPI provider"""
        client = NewsSentimentClient(provider="newsapi")
        assert client.provider_name == "newsapi"
        assert isinstance(client.provider, NewsAPIProvider)
    
    @pytest.mark.asyncio
    async def test_init_invalid_provider(self, mock_settings):
        """Test initialization with invalid provider"""
        with pytest.raises(ValueError, match="Unknown provider"):
            NewsSentimentClient(provider="invalid")
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_settings):
        """Test async context manager"""
        client = NewsSentimentClient()
        
        # Mock provider methods
        client.provider.__aenter__ = AsyncMock(return_value=client.provider)
        client.provider.__aexit__ = AsyncMock(return_value=None)
        
        async with client as c:
            assert c == client
        
        client.provider.__aenter__.assert_called_once()
        client.provider.__aexit__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_news_delegates(self, mock_settings):
        """Test that get_news delegates to provider"""
        client = NewsSentimentClient()
        
        # Mock provider method
        mock_articles = [Mock(spec=NewsArticle), Mock(spec=NewsArticle)]
        client.provider.get_news = AsyncMock(return_value=mock_articles)
        
        # Call get_news
        result = await client.get_news(
            symbols=["AAPL"],
            from_date=date(2025, 7, 1),
            to_date=date(2025, 7, 30),
            limit=50
        )
        
        # Verify delegation
        assert result == mock_articles
        client.provider.get_news.assert_called_once_with(
            symbols=["AAPL"],
            keywords=None,
            from_date=date(2025, 7, 1),
            to_date=date(2025, 7, 30),
            limit=50
        )
    
    @pytest.mark.asyncio
    async def test_get_sentiment_delegates(self, mock_settings):
        """Test that get_sentiment delegates to provider"""
        client = NewsSentimentClient()
        
        # Mock provider method
        mock_sentiments = [Mock(spec=NewsSentiment), Mock(spec=NewsSentiment)]
        client.provider.get_sentiment = AsyncMock(return_value=mock_sentiments)
        
        # Call get_sentiment
        result = await client.get_sentiment(
            symbols=["AAPL"],
            from_date=date(2025, 7, 1),
            to_date=date(2025, 7, 30),
            limit=50
        )
        
        # Verify delegation
        assert result == mock_sentiments
        client.provider.get_sentiment.assert_called_once_with(
            symbols=["AAPL"],
            from_date=date(2025, 7, 1),
            to_date=date(2025, 7, 30),
            limit=50
        )
    
    @pytest.mark.asyncio
    async def test_get_multi_symbol_sentiment_alphavantage(self, mock_settings):
        """Test multi-symbol sentiment for AlphaVantage (single symbol at a time)"""
        client = NewsSentimentClient(provider="alphavantage")
        
        # Create mock sentiments with proper article structure
        mock_article1 = Mock()
        mock_article1.symbols = ["AAPL"]
        aapl_sentiments = [Mock(spec=NewsSentiment)]
        aapl_sentiments[0].article = mock_article1
        
        mock_article2 = Mock()
        mock_article2.symbols = ["TSLA"]
        tsla_sentiments = [Mock(spec=NewsSentiment)]
        tsla_sentiments[0].article = mock_article2
        
        # Mock provider method to return different results
        client.provider.get_sentiment = AsyncMock(
            side_effect=[aapl_sentiments, tsla_sentiments]
        )
        
        # Mock asyncio.sleep to speed up test
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await client.get_multi_symbol_sentiment(
                symbols=["AAPL", "TSLA"],
                from_date=date(2025, 7, 1),
                limit_per_symbol=10
            )
        
        # Verify results
        assert "AAPL" in result
        assert "TSLA" in result
        assert result["AAPL"] == aapl_sentiments
        assert result["TSLA"] == tsla_sentiments
        
        # Verify two separate calls were made
        assert client.provider.get_sentiment.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_multi_symbol_sentiment_newsapi(self, mock_settings):
        """Test multi-symbol sentiment for NewsAPI (multiple symbols at once)"""
        client = NewsSentimentClient(provider="newsapi")
        
        # Mock sentiments with multiple symbols
        mock_article1 = Mock()
        mock_article1.symbols = ["AAPL", "MSFT"]
        mock_article2 = Mock()
        mock_article2.symbols = ["AAPL"]
        mock_article3 = Mock()
        mock_article3.symbols = ["TSLA"]
        
        mock_sentiments = [
            Mock(spec=NewsSentiment),
            Mock(spec=NewsSentiment),
            Mock(spec=NewsSentiment)
        ]
        mock_sentiments[0].article = mock_article1
        mock_sentiments[1].article = mock_article2
        mock_sentiments[2].article = mock_article3
        
        client.provider.get_sentiment = AsyncMock(return_value=mock_sentiments)
        
        result = await client.get_multi_symbol_sentiment(
            symbols=["AAPL", "TSLA", "MSFT"],
            limit_per_symbol=10
        )
        
        # Verify results are grouped by symbol
        assert "AAPL" in result
        assert "TSLA" in result
        assert "MSFT" in result
        assert len(result["AAPL"]) == 2  # Two articles mention AAPL
        assert len(result["TSLA"]) == 1
        assert len(result["MSFT"]) == 1
        
        # Verify only one call was made (NewsAPI supports multiple symbols)
        client.provider.get_sentiment.assert_called_once()


class TestNewsModels:
    """Test news sentiment models"""
    
    def test_news_article_creation(self):
        """Test NewsArticle model creation"""
        article = NewsArticle(
            source="Bloomberg",
            source_category=NewsSource.OTHER,
            title="Test Article",
            url="https://example.com/article",
            published_at=datetime.now()
        )
        
        assert article.source == "Bloomberg"
        assert article.title == "Test Article"
        assert len(article.article_id) > 0  # UUID generated
        assert article.symbols == []
        assert article.entities == []
        assert article.categories == []
    
    def test_sentiment_score_validation(self):
        """Test SentimentScore validation"""
        score = SentimentScore(
            polarity=0.5,
            subjectivity=0.7,
            positive=0.6,
            negative=0.1,
            neutral=0.3,
            bullish=0.5,
            bearish=None
        )
        
        assert score.polarity == 0.5
        assert score.subjectivity == 0.7
        assert score.positive == 0.6
        assert score.bullish == 0.5
        assert score.bearish is None
    
    def test_news_sentiment_tradeable_score(self):
        """Test tradeable score calculation"""
        article = NewsArticle(
            source="Test",
            source_category=NewsSource.OTHER,
            title="Test",
            url="https://example.com",
            published_at=datetime.now()
        )
        
        sentiment_score = SentimentScore(
            polarity=0.8,  # Strong positive
            subjectivity=0.5,
            positive=0.8,
            negative=0.0,
            neutral=0.2
        )
        
        sentiment = NewsSentiment(
            article_id=article.article_id,
            article=article,
            overall_sentiment=sentiment_score,
            sentiment_model="test",
            relevance_score=0.9,
            impact_score=0.7
        )
        
        # Tradeable score = (|0.8| * 0.6 + 0.9 * 0.4) * 0.7
        expected = (0.8 * 0.6 + 0.9 * 0.4) * 0.7
        assert abs(sentiment.get_tradeable_score() - expected) < 0.01