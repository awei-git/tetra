"""Generic data ingester for all data sources."""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import asyncio
from abc import ABC, abstractmethod

from sqlalchemy import select, and_, func
from sqlalchemy.dialects.postgresql import insert
from src.db import async_session_maker
from src.models.sqlalchemy import OHLCVModel
from src.models.sqlalchemy.economic_data import EconomicIndicatorModel
from src.models.sqlalchemy.news_sentiment import NewsArticleModel, NewsSentimentModel
from src.models.sqlalchemy.event_data import EventDataModel

logger = logging.getLogger(__name__)


class DataIngester:
    """
    Central data ingestion manager for all data types.
    
    This is the single entry point for all data ingestion into the database.
    Supports multiple data providers and data types.
    """
    
    def __init__(self, provider: str = "polygon"):
        """
        Initialize data ingester with specified provider.
        
        Args:
            provider: Data provider name (polygon, yfinance, alphavantage, fred, etc.)
        """
        self.provider = provider
        self._provider_instance = None
        self._initialize_provider()
        
    def _initialize_provider(self):
        """Initialize the appropriate provider based on configuration."""
        from .providers import (
            PolygonProvider, YFinanceProvider, AlphaVantageProvider,
            FREDProvider, NewsAPIProvider
        )
        
        provider_map = {
            'polygon': PolygonProvider,
            'yfinance': YFinanceProvider,
            'alphavantage': AlphaVantageProvider,
            'fred': FREDProvider,
            'newsapi': NewsAPIProvider
        }
        
        provider_class = provider_map.get(self.provider)
        if provider_class:
            self._provider_instance = provider_class()
        else:
            logger.warning(f"Unknown provider {self.provider}, using default")
            self._provider_instance = PolygonProvider()
    
    # ==================== MARKET DATA ====================
    
    async def ingest_ohlcv_batch(
        self,
        symbols: List[str],
        from_date: Union[date, datetime],
        to_date: Union[date, datetime],
        timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """
        Ingest OHLCV data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            from_date: Start date for data
            to_date: End date for data
            timeframe: Timeframe (1d, 1h, 5m, etc.)
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Ingesting OHLCV data for {len(symbols)} symbols from {from_date} to {to_date}")
        
        results = {
            "success": {},
            "failed": {},
            "total_records": 0,
            "symbols_processed": 0,
            "errors": 0
        }
        
        # Process in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            for symbol in batch:
                try:
                    # Fetch data from provider
                    data = await self._provider_instance.fetch_ohlcv(
                        symbol=symbol,
                        from_date=from_date,
                        to_date=to_date,
                        timeframe=timeframe
                    )
                    
                    if data:
                        # Store in database
                        records_stored = await self._store_ohlcv_data(symbol, data, timeframe)
                        results["success"][symbol] = records_stored
                        results["total_records"] += records_stored
                        results["symbols_processed"] += 1
                        logger.debug(f"Stored {records_stored} records for {symbol}")
                    else:
                        results["failed"][symbol] = "No data returned"
                        results["errors"] += 1
                        
                except Exception as e:
                    logger.error(f"Error ingesting {symbol}: {e}", exc_info=True)
                    results["failed"][symbol] = str(e)
                    results["errors"] += 1
            
            # Rate limiting
            await asyncio.sleep(0.1)
        
        logger.info(f"OHLCV ingestion complete: {results['symbols_processed']} symbols, {results['total_records']} records")
        return results
    
    async def _store_ohlcv_data(self, symbol: str, data: List[Dict], timeframe: str = "1d") -> int:
        """Store OHLCV data in database."""
        if not data:
            return 0
            
        async with async_session_maker() as session:
            records_stored = 0
            
            for record in data:
                stmt = insert(OHLCVModel).values(
                    symbol=symbol,
                    timestamp=record['timestamp'],
                    open=record['open'],
                    high=record['high'],
                    low=record['low'],
                    close=record['close'],
                    volume=record['volume'],
                    vwap=record.get('vwap'),
                    trades_count=record.get('trade_count'),
                    timeframe=timeframe,
                    source=self.provider
                )
                
                # On conflict, update the existing record
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'timestamp', 'timeframe'],
                    set_={
                        'open': stmt.excluded.open,
                        'high': stmt.excluded.high,
                        'low': stmt.excluded.low,
                        'close': stmt.excluded.close,
                        'volume': stmt.excluded.volume,
                        'vwap': stmt.excluded.vwap,
                        'trades_count': stmt.excluded.trades_count,
                        'source': stmt.excluded.source
                    }
                )
                
                await session.execute(stmt)
                records_stored += 1
            
            await session.commit()
            
        return records_stored
    
    # ==================== ECONOMIC DATA ====================
    
    async def ingest_economic_indicators(
        self,
        indicators: List[str],
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Ingest economic indicator data.
        
        Args:
            indicators: List of indicator symbols (GDP, CPI, etc.)
            from_date: Start date (optional)
            to_date: End date (optional)
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Ingesting economic indicators: {indicators}")
        
        results = {
            "success": {},
            "failed": {},
            "total_records": 0
        }
        
        for indicator in indicators:
            try:
                # Fetch data from provider (e.g., FRED)
                data = await self._provider_instance.fetch_economic_indicator(
                    indicator=indicator,
                    from_date=from_date,
                    to_date=to_date
                )
                
                if data:
                    # Store in database
                    records_stored = await self._store_economic_data(indicator, data)
                    results["success"][indicator] = records_stored
                    results["total_records"] += records_stored
                else:
                    results["failed"][indicator] = "No data returned"
                    
            except Exception as e:
                logger.error(f"Error ingesting {indicator}: {e}")
                results["failed"][indicator] = str(e)
        
        logger.info(f"Economic data ingestion complete: {results['total_records']} records")
        return results
    
    async def _store_economic_data(self, indicator: str, data: List[Dict]) -> int:
        """Store economic indicator data in database."""
        if not data:
            return 0
            
        async with async_session_maker() as session:
            records_stored = 0
            
            for record in data:
                stmt = insert(EconomicIndicatorModel).values(
                    symbol=indicator,
                    date=record['date'],
                    value=record['value'],
                    previous_value=record.get('previous_value'),
                    period=record.get('period'),
                    unit=record.get('unit')
                )
                
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'date'],
                    set_={
                        'value': stmt.excluded.value,
                        'previous_value': stmt.excluded.previous_value,
                        'updated_at': func.now()
                    }
                )
                
                await session.execute(stmt)
                records_stored += 1
            
            await session.commit()
            
        return records_stored
    
    # ==================== NEWS DATA ====================
    
    async def ingest_news(
        self,
        symbols: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Ingest news articles and sentiment.
        
        Args:
            symbols: List of symbols to get news for
            from_date: Start date
            to_date: End date
            categories: News categories to filter
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Ingesting news for symbols: {symbols}")
        
        results = {
            "articles": 0,
            "sentiments": 0,
            "errors": 0
        }
        
        try:
            # Fetch news from provider
            articles = await self._provider_instance.fetch_news(
                symbols=symbols,
                from_date=from_date,
                to_date=to_date,
                categories=categories
            )
            
            for article in articles:
                try:
                    # Store article
                    article_id = await self._store_news_article(article)
                    results["articles"] += 1
                    
                    # Calculate and store sentiment
                    if 'content' in article:
                        sentiment = await self._calculate_sentiment(article['content'])
                        await self._store_sentiment(article_id, sentiment)
                        results["sentiments"] += 1
                        
                except Exception as e:
                    logger.error(f"Error storing article: {e}")
                    results["errors"] += 1
                    
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            results["errors"] += 1
        
        logger.info(f"News ingestion complete: {results['articles']} articles, {results['sentiments']} sentiments")
        return results
    
    async def _store_news_article(self, article: Dict) -> int:
        """Store news article in database."""
        async with async_session_maker() as session:
            stmt = insert(NewsArticleModel).values(
                source=article.get('source'),
                author=article.get('author'),
                title=article['title'],
                description=article.get('description'),
                url=article['url'],
                published_at=article['published_at'],
                content=article.get('content'),
                symbols=article.get('symbols', [])
            )
            
            stmt = stmt.on_conflict_do_update(
                index_elements=['url'],
                set_={
                    'content': stmt.excluded.content,
                    'updated_at': func.now()
                }
            )
            
            result = await session.execute(stmt)
            await session.commit()
            
            # Get the article ID
            article_result = await session.execute(
                select(NewsArticleModel).where(NewsArticleModel.url == article['url'])
            )
            article_obj = article_result.scalar_one()
            
            return article_obj.article_id
    
    async def _calculate_sentiment(self, content: str) -> Dict[str, float]:
        """Calculate sentiment scores for content."""
        # This would use a sentiment analysis model
        # For now, return placeholder scores
        return {
            'positive': 0.3,
            'negative': 0.2,
            'neutral': 0.5,
            'compound': 0.1
        }
    
    async def _store_sentiment(self, article_id: int, sentiment: Dict):
        """Store sentiment scores in database."""
        async with async_session_maker() as session:
            stmt = insert(NewsSentimentModel).values(
                article_id=article_id,
                positive_score=sentiment['positive'],
                negative_score=sentiment['negative'],
                neutral_score=sentiment['neutral'],
                compound_score=sentiment['compound'],
                analyzed_at=datetime.now()
            )
            
            stmt = stmt.on_conflict_do_nothing()
            
            await session.execute(stmt)
            await session.commit()
    
    # ==================== EVENT DATA ====================
    
    async def ingest_events(
        self,
        event_types: List[str],
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Ingest event data (earnings, dividends, splits, etc.).
        
        Args:
            event_types: Types of events to ingest
            from_date: Start date
            to_date: End date
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Ingesting events: {event_types}")
        
        results = {
            "success": {},
            "failed": {},
            "total_records": 0
        }
        
        for event_type in event_types:
            try:
                # Fetch events from provider
                events = await self._provider_instance.fetch_events(
                    event_type=event_type,
                    from_date=from_date,
                    to_date=to_date
                )
                
                if events:
                    # Store in database
                    records_stored = await self._store_events(event_type, events)
                    results["success"][event_type] = records_stored
                    results["total_records"] += records_stored
                else:
                    results["failed"][event_type] = "No data returned"
                    
            except Exception as e:
                logger.error(f"Error ingesting {event_type}: {e}")
                results["failed"][event_type] = str(e)
        
        logger.info(f"Event ingestion complete: {results['total_records']} records")
        return results
    
    async def _store_events(self, event_type: str, events: List[Dict]) -> int:
        """Store event data in database."""
        if not events:
            return 0
            
        async with async_session_maker() as session:
            records_stored = 0
            
            for event in events:
                stmt = insert(EventDataModel).values(
                    symbol=event['symbol'],
                    event_type=event_type,
                    event_date=event['date'],
                    event_time=event.get('time'),
                    data=event.get('data', {}),
                    importance=event.get('importance', 'medium')
                )
                
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'event_type', 'event_date'],
                    set_={
                        'data': stmt.excluded.data,
                        'importance': stmt.excluded.importance,
                        'updated_at': func.now()
                    }
                )
                
                await session.execute(stmt)
                records_stored += 1
            
            await session.commit()
            
        return records_stored
    
    # ==================== BULK OPERATIONS ====================
    
    async def ingest_all_data_for_symbol(
        self,
        symbol: str,
        from_date: date,
        to_date: date
    ) -> Dict[str, Any]:
        """
        Ingest all available data types for a symbol.
        
        Args:
            symbol: Ticker symbol
            from_date: Start date
            to_date: End date
            
        Returns:
            Dictionary with results for each data type
        """
        logger.info(f"Ingesting all data for {symbol}")
        
        results = {}
        
        # OHLCV data
        results['ohlcv'] = await self.ingest_ohlcv_batch(
            symbols=[symbol],
            from_date=from_date,
            to_date=to_date
        )
        
        # News
        results['news'] = await self.ingest_news(
            symbols=[symbol],
            from_date=from_date,
            to_date=to_date
        )
        
        # Events
        results['events'] = await self.ingest_events(
            event_types=['earnings', 'dividends', 'splits'],
            from_date=from_date,
            to_date=to_date
        )
        
        return results
    
    async def close(self):
        """Close provider connections."""
        if self._provider_instance:
            await self._provider_instance.close()