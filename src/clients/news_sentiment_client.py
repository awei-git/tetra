"""News and sentiment data client supporting multiple providers"""

from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Protocol
from decimal import Decimal
import asyncio
from abc import ABC, abstractmethod

from config import settings
from src.clients.base_client import BaseAPIClient, RateLimiter
from src.models import (
    NewsArticle, SentimentScore, NewsSentiment, NewsSource, 
    SentimentType, NewsCategory
)
from src.utils.logging import logger


class NewsProvider(Protocol):
    """Protocol for news data providers"""
    
    async def get_news(
        self,
        symbols: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        limit: int = 100,
        **kwargs
    ) -> List[NewsArticle]:
        """Get news articles"""
        ...
    
    async def get_sentiment(
        self,
        symbols: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        limit: int = 100,
        **kwargs
    ) -> List[NewsSentiment]:
        """Get news with sentiment scores"""
        ...


class NewsAPIProvider(BaseAPIClient):
    """NewsAPI.org provider for news articles"""
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or settings.news_api_key
        if not api_key:
            raise ValueError("NewsAPI key not provided")
        
        rate_limiter = RateLimiter(
            calls=500,  # NewsAPI allows 500 requests per day for free tier
            period=86400  # 24 hours
        )
        
        super().__init__(
            base_url="https://newsapi.org/v2",
            api_key=None,  # NewsAPI uses key in params
            rate_limiter=rate_limiter,
            timeout=30
        )
        
        self.api_key = api_key
    
    def _add_api_key(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add API key to parameters"""
        if params is None:
            params = {}
        params["apiKey"] = self.api_key
        return params
    
    async def get_news(
        self,
        symbols: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        limit: int = 100,
        language: str = "en",
        sort_by: str = "relevancy"
    ) -> List[NewsArticle]:
        """Get news articles from NewsAPI"""
        
        # Build query
        query_parts = []
        if symbols:
            # Add stock symbols and company names
            symbol_queries = []
            for symbol in symbols:
                symbol_queries.append(f'"{symbol}"')
                # Could map symbols to company names here
            query_parts.append(f"({' OR '.join(symbol_queries)})")
        
        if keywords:
            query_parts.append(f"({' OR '.join(keywords)})")
        
        # Add financial keywords if no specific query
        if not query_parts:
            query_parts.append("(stock OR earnings OR market OR trading)")
        
        query = " AND ".join(query_parts)
        
        # Prepare parameters
        params = self._add_api_key({
            "q": query,
            "language": language,
            "sortBy": sort_by,
            "pageSize": min(limit, 100)  # NewsAPI max is 100
        })
        
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()
        
        # Make request
        try:
            response = await self.get("/everything", params=params)
            
            if response.get("status") != "ok":
                logger.error(f"NewsAPI error: {response}")
                return []
            
            articles = []
            for article_data in response.get("articles", []):
                # Extract symbols from title and description
                text = f"{article_data.get('title', '')} {article_data.get('description', '')}"
                detected_symbols = []
                if symbols:
                    for symbol in symbols:
                        if symbol.upper() in text.upper():
                            detected_symbols.append(symbol)
                
                article = NewsArticle(
                    source=article_data.get("source", {}).get("name", "Unknown"),
                    source_category=NewsSource.NEWSAPI,
                    author=article_data.get("author"),
                    title=article_data.get("title", ""),
                    description=article_data.get("description"),
                    content=article_data.get("content"),
                    url=article_data.get("url", ""),
                    image_url=article_data.get("urlToImage"),
                    published_at=datetime.fromisoformat(
                        article_data.get("publishedAt", "").replace("Z", "+00:00")
                    ),
                    symbols=detected_symbols,
                    raw_data=article_data
                )
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from NewsAPI")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching NewsAPI articles: {e}")
            return []
    
    async def get_sentiment(self, **kwargs) -> List[NewsSentiment]:
        """NewsAPI doesn't provide sentiment - would need to analyze separately"""
        logger.warning("NewsAPI doesn't provide sentiment scores directly")
        return []


class AlphaVantageNewsProvider(BaseAPIClient):
    """Alpha Vantage provider for news with sentiment"""
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or settings.alphavantage_api_key
        if not api_key:
            raise ValueError("Alpha Vantage API key not provided")
        
        rate_limiter = RateLimiter(
            calls=5,  # Alpha Vantage free tier: 5 calls per minute
            period=60
        )
        
        super().__init__(
            base_url="https://www.alphavantage.co",
            api_key=None,  # Alpha Vantage uses key in params
            rate_limiter=rate_limiter,
            timeout=30
        )
        
        self.api_key = api_key
    
    async def get_news(self, symbols: Optional[List[str]] = None, **kwargs) -> List[NewsArticle]:
        """Get news articles - Alpha Vantage requires symbol"""
        if not symbols:
            logger.warning("Alpha Vantage requires symbols for news query")
            return []
        
        # Alpha Vantage returns news with sentiment, so we'll use get_sentiment
        sentiments = await self.get_sentiment(symbols=symbols[:1], **kwargs)  # API takes one symbol
        
        # Extract just the articles
        return [sentiment.article for sentiment in sentiments]
    
    async def get_sentiment(
        self,
        symbols: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        limit: int = 50,
        topics: Optional[List[str]] = None
    ) -> List[NewsSentiment]:
        """Get news with sentiment scores from Alpha Vantage"""
        
        if not symbols or len(symbols) == 0:
            logger.warning("Alpha Vantage requires at least one symbol")
            return []
        
        # Alpha Vantage news sentiment API
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbols[0],  # API only accepts one ticker at a time
            "apikey": self.api_key,
            "limit": min(limit, 1000)
        }
        
        # Add optional parameters
        if from_date:
            params["time_from"] = from_date.strftime("%Y%m%dT0000")
        if to_date:
            params["time_to"] = to_date.strftime("%Y%m%dT2359")
        
        if topics:
            # Alpha Vantage topics: earnings, ipo, mergers_and_acquisitions, 
            # financial_markets, economy_fiscal, economy_monetary, economy_macro, 
            # energy_transportation, finance, life_sciences, manufacturing, 
            # real_estate, retail_wholesale, technology
            params["topics"] = ",".join(topics)
        
        try:
            response = await self.get("/query", params=params)
            
            # Check if response is a string (error message)
            if isinstance(response, str):
                logger.error(f"Alpha Vantage returned string response: {response}")
                return []
            
            if "Error Message" in response:
                logger.error(f"Alpha Vantage error: {response['Error Message']}")
                return []
            
            if "Note" in response:
                logger.warning(f"Alpha Vantage note: {response['Note']}")
                return []
            
            if "Information" in response:
                logger.warning(f"Alpha Vantage rate limit: {response['Information']}")
                return []
            
            sentiments = []
            feed = response.get("feed", [])
            
            for item in feed:
                # Create article
                article = NewsArticle(
                    source_id=item.get("url"),
                    source=item.get("source", "Unknown"),
                    source_category=NewsSource.OTHER,
                    author=None,  # Not provided
                    title=item.get("title", ""),
                    description=item.get("summary"),
                    content=None,  # Not provided
                    url=item.get("url", ""),
                    image_url=item.get("banner_image"),
                    published_at=datetime.strptime(
                        item.get("time_published", ""), "%Y%m%dT%H%M%S"
                    ),
                    symbols=[symbols[0]],  # The symbol we queried
                    entities=[author.get("name", "") if isinstance(author, dict) else str(author) 
                              for author in item.get("authors", [])],
                    categories=self._map_topics_to_categories(item.get("topics", [])),
                    raw_data=item
                )
                
                # Extract sentiment scores
                ticker_sentiment = {}
                for ts in item.get("ticker_sentiment", []):
                    if ts.get("ticker") == symbols[0]:
                        ticker_sentiment = ts
                        break
                
                # Create sentiment score
                # Alpha Vantage provides: sentiment_score (-1 to 1), sentiment_score_label
                av_score = float(ticker_sentiment.get("ticker_sentiment_score", 0))
                av_label = ticker_sentiment.get("ticker_sentiment_label", "Neutral")
                
                sentiment_score = SentimentScore(
                    polarity=av_score,
                    subjectivity=0.5,  # Not provided, use neutral
                    positive=max(0, av_score) if av_score > 0.1 else 0,
                    negative=abs(min(0, av_score)) if av_score < -0.1 else 0,
                    neutral=1.0 if -0.1 <= av_score <= 0.1 else 0,
                    bullish=max(0, av_score) if av_score > 0 else None,
                    bearish=abs(min(0, av_score)) if av_score < 0 else None
                )
                
                # Create news sentiment
                sentiment = NewsSentiment(
                    article_id=article.article_id,
                    article=article,
                    sentiment_scores={
                        SentimentType.OVERALL: sentiment_score
                    },
                    overall_sentiment=sentiment_score,
                    sentiment_model="alphavantage",
                    relevance_score=float(ticker_sentiment.get("relevance_score", 0)),
                    impact_score=self._estimate_impact(av_score, av_label)
                )
                
                sentiments.append(sentiment)
            
            logger.info(f"Fetched {len(sentiments)} news items with sentiment from Alpha Vantage")
            return sentiments
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return []
    
    def _map_topics_to_categories(self, topics: List[Dict]) -> List[NewsCategory]:
        """Map Alpha Vantage topics to our categories"""
        category_map = {
            "earnings": NewsCategory.EARNINGS,
            "mergers_and_acquisitions": NewsCategory.MERGER,
            "financial_markets": NewsCategory.MARKET,
            "economy": NewsCategory.MACRO,
            "technology": NewsCategory.OTHER,
        }
        
        categories = []
        for topic in topics:
            topic_name = topic.get("topic", "").lower()
            for key, category in category_map.items():
                if key in topic_name:
                    categories.append(category)
                    break
        
        return categories or [NewsCategory.OTHER]
    
    def _estimate_impact(self, score: float, label: str) -> float:
        """Estimate impact score based on sentiment strength"""
        abs_score = abs(score)
        if abs_score > 0.5:
            return 0.8
        elif abs_score > 0.3:
            return 0.6
        elif abs_score > 0.1:
            return 0.4
        else:
            return 0.2


class NewsSentimentClient:
    """Main client for news and sentiment data with support for multiple providers"""
    
    def __init__(self, provider: str = "alphavantage"):
        """Initialize with specified provider"""
        self.provider_name = provider.lower()
        self.provider = self._create_provider(provider)
    
    def _create_provider(self, provider: str) -> NewsProvider:
        """Create provider instance based on name"""
        provider = provider.lower()
        
        if provider == "newsapi":
            return NewsAPIProvider()
        elif provider in ["alphavantage", "alpha_vantage", "av"]:
            return AlphaVantageNewsProvider()
        else:
            raise ValueError(f"Unknown provider: {provider}. Available: newsapi, alphavantage")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.provider.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.provider.__aexit__(exc_type, exc_val, exc_tb)
    
    async def get_news(
        self,
        symbols: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        limit: int = 100,
        **kwargs
    ) -> List[NewsArticle]:
        """Get news articles"""
        return await self.provider.get_news(
            symbols=symbols,
            keywords=keywords,
            from_date=from_date,
            to_date=to_date,
            limit=limit,
            **kwargs
        )
    
    async def get_sentiment(
        self,
        symbols: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        limit: int = 100,
        **kwargs
    ) -> List[NewsSentiment]:
        """Get news with sentiment scores"""
        return await self.provider.get_sentiment(
            symbols=symbols,
            from_date=from_date,
            to_date=to_date,
            limit=limit,
            **kwargs
        )
    
    async def get_multi_symbol_sentiment(
        self,
        symbols: List[str],
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        limit_per_symbol: int = 50
    ) -> Dict[str, List[NewsSentiment]]:
        """Get sentiment for multiple symbols (handles provider limitations)"""
        results = {}
        
        # Some providers (like AlphaVantage) only support one symbol at a time
        if self.provider_name in ["alphavantage", "alpha_vantage", "av"]:
            for symbol in symbols:
                logger.info(f"Fetching sentiment for {symbol}")
                sentiments = await self.get_sentiment(
                    symbols=[symbol],
                    from_date=from_date,
                    to_date=to_date,
                    limit=limit_per_symbol
                )
                results[symbol] = sentiments
                
                # Add delay between requests to respect rate limits
                if len(symbols) > 1:
                    await asyncio.sleep(12)  # Alpha Vantage: 5 req/min
        else:
            # Providers that support multiple symbols
            sentiments = await self.get_sentiment(
                symbols=symbols,
                from_date=from_date,
                to_date=to_date,
                limit=limit_per_symbol * len(symbols)
            )
            
            # Group by symbol
            for sentiment in sentiments:
                for symbol in sentiment.article.symbols:
                    if symbol not in results:
                        results[symbol] = []
                    results[symbol].append(sentiment)
        
        return results