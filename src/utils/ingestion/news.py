"""News data ingestion helpers."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

import httpx
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.schema import news_articles
from src.db.session import engine
from src.definitions.market_universe import MarketUniverse
from src.utils.ingestion.clients import FinnhubClient, NewsAPIClient
from src.utils.ingestion.common import chunk
from src.utils.ingestion.sentiment import analyze_sentiment, normalize_sentiment
from src.utils.ingestion.topics import detect_macro_topics
from src.utils.ingestion.types import IngestionSummary

UTC = timezone.utc


async def ingest_news_data(
    start: date,
    end: date,
    query: Optional[str] = None,
    symbols: Optional[List[str]] = None,
) -> IngestionSummary:
    query = query or "stocks"
    rows: List[Dict] = []

    finnhub_client: Optional[FinnhubClient] = None
    newsapi_client: Optional[NewsAPIClient] = None

    try:
        finnhub_client = FinnhubClient()
    except RuntimeError as exc:
        print(f"Finnhub news disabled: {exc}")
    try:
        newsapi_client = NewsAPIClient()
    except RuntimeError as exc:
        print(f"NewsAPI disabled: {exc}")

    general_count = 0
    company_count = 0
    newsapi_count = 0

    if finnhub_client:
        general_news = await finnhub_client.get_general_news(limit=100)
        general_count = len(general_news)
        for article in general_news:
            published_at = article.get("datetime") or article.get("time")
            if published_at is None:
                published = datetime.now(tz=UTC)
            else:
                try:
                    published = datetime.fromtimestamp(float(published_at), tz=UTC)
                except Exception:
                    published = datetime.fromisoformat(str(published_at).replace(" ", "T"))
            external_id_raw = article.get("id") or article.get("url") or article.get("news_url")
            external_id = str(external_id_raw) if external_id_raw is not None else None
            headline = article.get("headline") or article.get("title") or ""
            summary = article.get("summary") or article.get("description")
            text = " ".join([headline, summary or ""]).strip()
            score, confidence, label = normalize_sentiment(article.get("sentimentScore"))
            if score is None:
                score, confidence, label = analyze_sentiment(text)
            topics = detect_macro_topics(text)
            payload = {**article, "analysis": {"sentiment": score, "label": label, "topics": topics}}
            rows.append(
                {
                    "external_id": external_id[:512] if external_id else None,
                    "headline": headline,
                    "summary": summary,
                    "url": article.get("url") or article.get("news_url"),
                    "source": article.get("source") or article.get("provider"),
                    "published_at": published,
                    "tickers": None,
                    "sentiment": score,
                    "sentiment_confidence": confidence,
                    "embeddings": {},
                    "payload": payload,
                    "ingested_at": datetime.now(tz=UTC),
                }
            )

        symbol_list = symbols or MarketUniverse.get_all_symbols()[:100]
        for symbol in symbol_list:
            try:
                articles = await finnhub_client.get_company_news(symbol, start, end)
            except httpx.HTTPStatusError as exc:
                print(f"Skipping Finnhub company news for {symbol}: {exc.response.status_code}")
                await asyncio.sleep(0.2)
                continue
            if not articles:
                await asyncio.sleep(0.2)
                continue
            company_count += len(articles)
            for article in articles:
                published_at = article.get("datetime")
                try:
                    published = datetime.fromtimestamp(float(published_at), tz=UTC)
                except Exception:
                    published = datetime.fromisoformat(str(published_at).replace(" ", "T"))
                external_id_raw = article.get("id") or article.get("url") or f"FH-{symbol}-{published.isoformat()}"
                external_id = str(external_id_raw) if external_id_raw is not None else None
                headline = article.get("headline") or article.get("title") or ""
                summary = article.get("summary") or article.get("description")
                text = " ".join([headline, summary or ""]).strip()
                score, confidence, label = analyze_sentiment(text)
                topics = detect_macro_topics(text)
                payload = {**article, "analysis": {"sentiment": score, "label": label, "topics": topics}}
                rows.append(
                    {
                        "external_id": external_id[:512] if external_id else None,
                        "headline": headline,
                        "summary": summary,
                        "url": article.get("url"),
                        "source": article.get("source") or article.get("provider"),
                        "published_at": published,
                        "tickers": [symbol],
                        "sentiment": score,
                        "sentiment_confidence": confidence,
                        "embeddings": {},
                        "payload": payload,
                        "ingested_at": datetime.now(tz=UTC),
                    }
                )
            await asyncio.sleep(0.2)

    if newsapi_client:
        try:
            articles = await newsapi_client.get_articles(query=query, start=start, end=end, max_pages=2)
        except httpx.HTTPStatusError as exc:
            print(f"Skipping NewsAPI articles: {exc.response.status_code}")
            articles = []
        newsapi_count = len(articles)
        for article in articles:
            published_at = article.get("publishedAt")
            if not published_at:
                continue
            published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            external_id_raw = article.get("url")
            external_id = str(external_id_raw) if external_id_raw is not None else None
            headline = article.get("title") or ""
            summary = article.get("description")
            text = " ".join([headline, summary or ""]).strip()
            score, confidence, label = analyze_sentiment(text)
            topics = detect_macro_topics(text)
            payload = {**article, "analysis": {"sentiment": score, "label": label, "topics": topics}}
            rows.append(
                {
                    "external_id": external_id[:512] if external_id else None,
                    "headline": headline,
                    "summary": summary,
                    "url": external_id,
                    "source": (article.get("source") or {}).get("name"),
                    "published_at": published,
                    "tickers": None,
                    "sentiment": score,
                    "sentiment_confidence": confidence,
                    "embeddings": {},
                    "payload": payload,
                    "ingested_at": datetime.now(tz=UTC),
                }
            )

    deduped = rows
    if rows:
        dedup = {}
        for row in rows:
            key = (row.get("source"), row.get("external_id"), row.get("published_at"))
            if key in dedup:
                continue
            dedup[key] = row
        deduped = list(dedup.values())
        async with engine.begin() as conn:
            for batch in chunk(deduped):
                stmt = pg_insert(news_articles).values(batch)
                await conn.execute(
                    stmt.on_conflict_do_update(
                        index_elements=[news_articles.c.source, news_articles.c.external_id, news_articles.c.published_at],
                        set_={
                            "headline": stmt.excluded.headline,
                            "summary": stmt.excluded.summary,
                            "url": stmt.excluded.url,
                            "tickers": stmt.excluded.tickers,
                            "payload": stmt.excluded.payload,
                            "ingested_at": stmt.excluded.ingested_at,
                        },
                    )
                )

    if finnhub_client:
        await finnhub_client.close()
    if newsapi_client:
        await newsapi_client.close()

    return IngestionSummary(
        records=len(deduped),
        details={
            "finnhub_general": general_count,
            "finnhub_company": company_count,
            "newsapi": newsapi_count,
            "deduped": len(deduped),
        },
    )
