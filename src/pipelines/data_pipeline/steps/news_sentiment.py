"""News and sentiment data pipeline step"""

from datetime import date, timedelta
from typing import Dict, Any
import asyncio

from src.pipelines.base import PipelineStep, PipelineContext
from src.clients.news_sentiment_client import NewsSentimentClient
from src.db.base import get_session
from src.utils.logging import logger
from src.models.sqlalchemy.news_sentiment import NewsArticleModel
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime, timezone


class NewsSentimentStep(PipelineStep[Dict[str, Any]]):
    """
    Step for fetching news and sentiment data.
    Due to API limitations, this step may process a subset of symbols.
    """
    
    def __init__(self):
        super().__init__(
            name="NewsSentimentStep",
            description="Fetch news articles and sentiment analysis"
        )
        
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Execute news sentiment fetching"""
        mode = context.data.get("mode", "daily")
        provider = context.data.get("news_provider", "newsapi" if mode == "daily" else "alphavantage")
        
        # Use news-specific symbol list if available, otherwise use main symbols
        symbols = context.data.get("news_symbols", context.data.get("symbols", []))
        
        start_date = context.data.get("start_date")
        end_date = context.data.get("end_date")
        
        # For news, we typically look back a bit
        if mode == "daily":
            fetch_start = start_date - timedelta(days=2)
        else:
            fetch_start = start_date
            
        logger.info(
            f"Starting news sentiment {mode} for {len(symbols)} symbols using {provider}"
        )
        
        results = {
            "articles": 0,
            "sentiments": 0,
            "symbols_processed": 0,
            "errors": []
        }
        
        async with NewsSentimentClient(provider=provider) as client:
                
                if provider == "newsapi":
                    # NewsAPI can handle multiple symbols in one request
                    try:
                        # Process in chunks to avoid query length limits
                        chunk_size = 10
                        for i in range(0, len(symbols), chunk_size):
                            chunk = symbols[i:i + chunk_size]
                            
                            articles = await client.get_news(
                                symbols=chunk,
                                from_date=fetch_start,
                                to_date=end_date,
                                limit=100
                            )
                            
                            # Save articles to database
                            if articles:
                                async for db in get_session():
                                    await self._save_articles(articles, db)
                                    break
                            
                            results["articles"] += len(articles)
                            results["symbols_processed"] += len(chunk)
                            
                            await asyncio.sleep(0.5)  # Rate limiting
                            
                    except Exception as e:
                        error_msg = f"NewsAPI fetch failed: {str(e)}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)
                        context.add_error(error_msg)
                        
                else:  # AlphaVantage
                    # AlphaVantage requires one symbol at a time and has strict rate limits
                    # For daily mode, limit to most important symbols
                    if mode == "daily":
                        process_symbols = symbols[:5]  # Top 5 only for daily
                        logger.info(f"AlphaVantage daily mode: limiting to {len(process_symbols)} symbols")
                    else:
                        process_symbols = symbols
                        
                    for symbol in process_symbols:
                        try:
                            sentiments = await client.get_sentiment(
                                symbols=[symbol],
                                from_date=fetch_start,
                                to_date=end_date,
                                limit=50 if mode == "daily" else 100
                            )
                            
                            # For now, just count sentiments
                            results["sentiments"] += len(sentiments)
                            results["articles"] += len(sentiments)  # Each sentiment has an article
                            results["symbols_processed"] += 1
                            
                            # AlphaVantage rate limit: 5 requests per minute
                            await asyncio.sleep(12)
                            
                        except Exception as e:
                            error_msg = f"Failed to fetch news for {symbol}: {str(e)}"
                            logger.error(error_msg)
                            results["errors"].append(error_msg)
                            
                            # Check for rate limit errors
                            if "rate limit" in str(e).lower():
                                logger.warning("Rate limit hit, stopping news fetch")
                                break
        
        # Update metrics
        context.set_metric("news_articles", results["articles"])
        context.set_metric("news_sentiments", results["sentiments"])
        context.set_metric("news_symbols_processed", results["symbols_processed"])
        
        logger.info(
            f"News sentiment complete: {results['symbols_processed']} symbols processed, "
            f"{results['articles']} articles, {results['sentiments']} sentiments"
        )
        
        return results
    
    async def _save_articles(self, articles, db):
        """Save news articles to database with upsert logic"""
        try:
            # Prepare values for insert
            values = []
            for article in articles:
                values.append({
                    "url": article.url,
                    "title": article.title,
                    "author": article.author,
                    "source": article.source,
                    "source_category": "news",
                    "published_at": article.published_at,
                    "content": article.content,
                    "description": getattr(article, 'description', None),
                    "fetched_at": datetime.now(timezone.utc),
                    "symbols": article.symbols if hasattr(article, 'symbols') else [],  # List of symbols
                })
            
            # Batch insert to avoid parameter limits
            batch_size = 100
            for i in range(0, len(values), batch_size):
                batch = values[i:i + batch_size]
                
                # Use PostgreSQL upsert
                stmt = insert(NewsArticleModel).values(batch)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_news_articles_url",
                    set_={
                        "title": stmt.excluded.title,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
                
                await db.execute(stmt)
            
            await db.commit()
            logger.info(f"Saved {len(articles)} news articles to database")
            
        except Exception as e:
            logger.error(f"Error saving news articles: {e}")
            await db.rollback()
            raise