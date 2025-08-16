"""NewsAPI data provider implementation."""

import os
import yaml
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import aiohttp
import asyncio

from .base import BaseProvider

logger = logging.getLogger(__name__)


class NewsAPIProvider(BaseProvider):
    """
    NewsAPI.org data provider for news articles.
    
    Supports:
    - News articles from thousands of sources
    - Search by keywords, sources, domains
    - Category filtering (business, technology, etc.)
    - Sentiment analysis (if processed separately)
    
    Note: Free tier limited to 100 requests per day.
    """
    
    BASE_URL = "https://newsapi.org/v2"
    
    # News categories
    CATEGORIES = [
        "business",
        "entertainment",
        "general",
        "health",
        "science",
        "sports",
        "technology"
    ]
    
    # Financial news sources
    FINANCIAL_SOURCES = [
        "bloomberg",
        "business-insider",
        "cnbc",
        "financial-times",
        "fortune",
        "the-wall-street-journal",
        "the-economist",
        "reuters",
        "techcrunch",
        "the-verge"
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NewsAPI provider.
        
        Args:
            api_key: NewsAPI key (or from secrets.yml)
        """
        if not api_key:
            # Read from secrets.yml
            try:
                with open('config/secrets.yml', 'r') as f:
                    secrets = yaml.safe_load(f)
                    api_key = secrets.get('api_keys', {}).get('news_api')
            except:
                api_key = os.getenv("NEWSAPI_KEY")
        
        if not api_key:
            logger.warning("No NewsAPI key provided, functionality will be limited")
        
        super().__init__(api_key)
    
    def _initialize(self):
        """Initialize provider."""
        pass
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict:
        """
        Make HTTP request to NewsAPI.
        
        Args:
            endpoint: API endpoint (/everything, /top-headlines)
            params: Query parameters
            
        Returns:
            JSON response data
        """
        session = await self._get_session()
        
        headers = {
            "X-Api-Key": self.api_key
        }
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Check for API errors
                if data.get("status") != "ok":
                    error_msg = data.get("message", "Unknown error")
                    raise ValueError(f"NewsAPI Error: {error_msg}")
                
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"NewsAPI request failed: {e}")
            raise
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        from_date: Union[date, datetime],
        to_date: Union[date, datetime],
        timeframe: str = "1d"
    ) -> List[Dict[str, Any]]:
        """
        NewsAPI doesn't provide OHLCV data.
        
        Use market data providers instead.
        """
        logger.warning("NewsAPI doesn't provide OHLCV data. Use market data providers.")
        return []
    
    async def fetch_economic_indicator(
        self,
        indicator: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        NewsAPI doesn't provide economic indicators.
        
        Use FRED or Alpha Vantage providers instead.
        """
        logger.warning("NewsAPI doesn't provide economic indicators. Use FRED provider.")
        return []
    
    async def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles from NewsAPI.
        
        Args:
            symbols: List of symbols/keywords to search for
            from_date: Start date (max 1 month ago for free tier)
            to_date: End date
            categories: News categories to filter
            
        Returns:
            List of news articles
        """
        # NewsAPI free tier only allows up to 1 month of history
        max_history = datetime.now().date() - timedelta(days=30)
        if from_date and from_date < max_history:
            logger.warning(f"NewsAPI free tier only allows 30 days history. Adjusting from_date to {max_history}")
            from_date = max_history
        
        # Build query
        query_parts = []
        
        if symbols:
            # Add symbols as keywords
            query_parts.extend(symbols)
        
        # Use everything endpoint for keyword search, top-headlines for categories
        if query_parts:
            endpoint = "/everything"
            params = {
                "q": " OR ".join(query_parts),
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100  # Max allowed
            }
            
            # Add financial sources for better relevance
            params["sources"] = ",".join(self.FINANCIAL_SOURCES)
            
        else:
            endpoint = "/top-headlines"
            params = {
                "country": "us",
                "pageSize": 100
            }
            
            if categories:
                # NewsAPI only supports one category at a time
                params["category"] = categories[0] if categories else "business"
        
        # Add date filters
        if from_date:
            params["from"] = from_date.isoformat()
        
        if to_date:
            params["to"] = to_date.isoformat()
        
        try:
            response = await self._make_request(endpoint, params)
            
            articles = response.get("articles", [])
            
            if not articles:
                logger.warning(f"No news articles found for query: {query_parts}")
                return []
            
            # Convert to standard format
            results = []
            for article in articles:
                # Parse published date
                published_str = article.get("publishedAt", "")
                if published_str:
                    published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                else:
                    published_at = datetime.now()
                
                results.append({
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "author": article.get("author"),
                    "title": article.get("title", ""),
                    "description": article.get("description"),
                    "url": article.get("url", ""),
                    "published_at": published_at,
                    "content": article.get("content"),  # Usually truncated
                    "symbols": symbols if symbols else []  # Associate with searched symbols
                })
            
            logger.info(f"Fetched {len(results)} news articles")
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")
            return []
    
    async def fetch_events(
        self,
        event_type: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        NewsAPI doesn't provide structured event data.
        
        Could potentially extract events from news content.
        """
        logger.warning(f"NewsAPI doesn't provide structured {event_type} events")
        
        # Could search for event-related news
        if event_type == "earnings":
            # Search for earnings-related news
            earnings_keywords = ["earnings report", "quarterly results", "Q1", "Q2", "Q3", "Q4"]
            news = await self.fetch_news(
                symbols=earnings_keywords,
                from_date=from_date,
                to_date=to_date,
                categories=["business"]
            )
            
            # Convert news to pseudo-events
            events = []
            for article in news:
                # Try to extract company name from title
                title_words = article["title"].split()
                
                # Simple heuristic to find ticker symbols
                potential_symbols = [w for w in title_words if w.isupper() and 2 <= len(w) <= 5]
                
                if potential_symbols:
                    events.append({
                        "symbol": potential_symbols[0],
                        "date": article["published_at"].date(),
                        "time": article["published_at"].strftime("%H:%M:%S"),
                        "data": {
                            "title": article["title"],
                            "url": article["url"],
                            "description": article["description"]
                        },
                        "importance": "medium"
                    })
            
            return events
        
        return []
    
    async def fetch_headlines(
        self,
        category: str = "business",
        country: str = "us",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fetch top headlines.
        
        Args:
            category: News category
            country: Country code (us, gb, etc.)
            limit: Maximum number of headlines
            
        Returns:
            List of headlines
        """
        params = {
            "country": country,
            "category": category,
            "pageSize": min(limit, 100)
        }
        
        try:
            response = await self._make_request("/top-headlines", params)
            
            articles = response.get("articles", [])
            
            results = []
            for article in articles:
                published_str = article.get("publishedAt", "")
                if published_str:
                    published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                else:
                    published_at = datetime.now()
                
                results.append({
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "author": article.get("author"),
                    "title": article.get("title", ""),
                    "description": article.get("description"),
                    "url": article.get("url", ""),
                    "published_at": published_at,
                    "content": article.get("content"),
                    "symbols": []
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch headlines: {e}")
            return []
    
    async def search_news(
        self,
        query: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        sort_by: str = "publishedAt"
    ) -> List[Dict[str, Any]]:
        """
        Search news by custom query.
        
        Args:
            query: Search query
            from_date: Start date
            to_date: End date
            sort_by: Sort order (relevancy, popularity, publishedAt)
            
        Returns:
            List of matching articles
        """
        params = {
            "q": query,
            "language": "en",
            "sortBy": sort_by,
            "pageSize": 100
        }
        
        if from_date:
            params["from"] = from_date.isoformat()
        
        if to_date:
            params["to"] = to_date.isoformat()
        
        try:
            response = await self._make_request("/everything", params)
            
            articles = response.get("articles", [])
            
            results = []
            for article in articles:
                published_str = article.get("publishedAt", "")
                if published_str:
                    published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                else:
                    published_at = datetime.now()
                
                results.append({
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "author": article.get("author"),
                    "title": article.get("title", ""),
                    "description": article.get("description"),
                    "url": article.get("url", ""),
                    "published_at": published_at,
                    "content": article.get("content"),
                    "symbols": []
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search news: {e}")
            return []