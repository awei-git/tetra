"""Pre-market comprehensive morning briefing.

Runs ~7:00 AM ET weekdays. Combines:
1. Market data (overnight news, polymarket, rates, portfolio, events, narrative)
2. Tech/social feeds (Arxiv, HN, Reddit, GitHub, RSS)
3. LLM synthesis — conversational narrative tying market + tech signals together

Outputs:
- PDF report (emailed)
- Markdown briefing (pushed to Mira app via artifacts)

Usage:
  python scripts/run_premarket_report.py
  python scripts/run_premarket_report.py --no-llm
  python scripts/run_premarket_report.py --llm-provider deepseek
  python scripts/run_premarket_report.py --skip-feeds
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import settings
from src.report.llm_clients import create_clients

UTC = timezone.utc
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comprehensive morning briefing")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM commentary")
    parser.add_argument(
        "--llm-provider", type=str, default=None,
        help="Preferred LLM provider (openai, deepseek, gemini, claude)",
    )
    parser.add_argument("--skip-ingest", action="store_true", help="Skip data refresh")
    parser.add_argument("--skip-feeds", action="store_true", help="Skip feed fetching")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation")
    parser.add_argument("--no-email", action="store_true", help="Skip email delivery")
    return parser.parse_args()


def get_llm_client(provider: str | None = None):
    clients = create_clients(settings)
    if not clients:
        return None
    if provider and provider in clients:
        return clients[provider]
    for name in ["deepseek", "openai", "gemini", "claude"]:
        if name in clients:
            logging.info(f"Using LLM provider: {name}")
            return clients[name]
    return None


# ── Data fetchers (pre-market safe — no market OHLCV needed) ──────────────

async def _ingest_premarket_data() -> Dict[str, Any]:
    """Run lightweight data ingestion: news + polymarket + economic only."""
    from src.pipelines.data.runner import run_all_pipelines

    today = date.today()
    yesterday = today - timedelta(days=1)

    results = await run_all_pipelines(
        start=yesterday,
        end=today,
        include_market=False,
        include_events=True,
        include_economic=True,
        include_news=True,
        include_fundamentals=False,
        include_polymarket=True,
    )

    summary = {}
    for r in results:
        summary[r.pipeline] = {"records": r.records, "details": r.details}
    return summary


async def _fetch_overnight_news(since_hours: int = 20) -> List[Dict[str, Any]]:
    from sqlalchemy import text
    from src.db.session import engine

    cutoff = datetime.now(tz=UTC) - timedelta(hours=since_hours)

    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT headline, source, published_at, tickers, sentiment
            FROM news.articles
            WHERE published_at >= :cutoff
            ORDER BY published_at DESC
            LIMIT 50
        """), {"cutoff": cutoff})
        rows = result.fetchall()

    articles = []
    for r in rows:
        articles.append({
            "title": r.headline,
            "source": r.source,
            "published_at": r.published_at.isoformat() if r.published_at else "",
            "symbols": list(r.tickers) if r.tickers else [],
            "sentiment": float(r.sentiment) if r.sentiment else 0,
        })
    return articles


async def _fetch_narrative_state(as_of: date) -> Optional[Dict[str, Any]]:
    from sqlalchemy import text
    from src.db.session import engine

    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT date, dominant_narrative, narrative_shift, shift_magnitude,
                   counter_narrative, novelty
            FROM narrative.daily_state
            WHERE scope = 'market' AND date >= :start
            ORDER BY date DESC LIMIT 1
        """), {"start": as_of - timedelta(days=3)})
        row = result.fetchone()

    if not row:
        return None
    return {
        "date": row.date.isoformat(),
        "dominant_narrative": row.dominant_narrative,
        "narrative_shift": round(float(row.narrative_shift), 3) if row.narrative_shift else 0,
        "shift_magnitude": round(float(row.shift_magnitude), 3) if row.shift_magnitude else 0,
        "counter_narrative": row.counter_narrative,
    }


async def _fetch_polymarket_signals() -> List[Dict[str, Any]]:
    from sqlalchemy import text
    from src.db.session import engine

    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT m.slug, m.question, m.best_bid, m.volume, s.snapshot_time
            FROM polymarket.markets m
            JOIN polymarket.snapshots s ON m.market_id = s.market_id
            WHERE s.snapshot_time >= NOW() - INTERVAL '24 hours'
              AND m.active = true
            ORDER BY m.volume DESC NULLS LAST
            LIMIT 15
        """))
        rows = result.fetchall()

    signals = []
    for r in rows:
        price = float(r.best_bid) if r.best_bid is not None else None
        question = (r.question[:100] if r.question else r.slug or "Unknown")
        if price is None:
            continue  # Skip markets with no price data
        signals.append({
            "market": r.slug or "",
            "question": question,
            "price": price,
            "change_24h": 0,
            "volume_24h": float(r.volume) if r.volume else 0,
        })
    return signals


async def _fetch_portfolio_snapshot() -> Dict[str, Any]:
    from sqlalchemy import text
    from src.db.session import engine

    data: Dict[str, Any] = {}

    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT symbol, shares, current_price, market_value, weight, unrealized_pnl
            FROM portfolio.positions
            ORDER BY market_value DESC NULLS LAST
        """))
        positions = result.fetchall()
        data["positions"] = [
            {
                "symbol": r.symbol,
                "price": float(r.current_price) if r.current_price else 0,
                "value": float(r.market_value) if r.market_value else 0,
                "weight": round(float(r.weight) * 100, 1) if r.weight else 0,
                "pnl": float(r.unrealized_pnl) if r.unrealized_pnl else 0,
            }
            for r in positions
        ]

        result = await conn.execute(text("""
            SELECT total_value, cash, daily_return, cumulative_return
            FROM portfolio.snapshots
            ORDER BY date DESC LIMIT 1
        """))
        snap = result.fetchone()
        if snap:
            data["summary"] = {
                "total_value": float(snap.total_value),
                "cash": float(snap.cash),
                "daily_return": float(snap.daily_return),
                "cumulative_return": float(snap.cumulative_return),
            }

    return data


async def _fetch_rates_snapshot() -> List[Dict[str, Any]]:
    from sqlalchemy import text
    from src.db.session import engine

    fred_labels = {
        "VIXCLS": "VIX",
        "DGS10": "10Y Treasury",
        "DGS2": "2Y Treasury",
        "T10Y2Y": "10Y-2Y Spread",
        "BAMLH0A0HYM2": "HY Credit Spread",
    }
    fred_ids = list(fred_labels.keys())

    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT s.series_id, v.value, v.timestamp
            FROM economic.values v
            JOIN economic.series s ON v.series_id = s.series_id
            WHERE s.series_id = ANY(:ids)
            ORDER BY v.timestamp DESC
        """), {"ids": fred_ids})
        rows = result.fetchall()

    seen: set = set()
    rates = []
    for r in rows:
        sid = r.series_id
        if sid not in seen:
            seen.add(sid)
            rates.append({
                "name": fred_labels.get(sid, sid),
                "value": float(r.value),
                "as_of": r.timestamp.isoformat() if hasattr(r.timestamp, "isoformat") else str(r.timestamp),
            })
    return rates


async def _fetch_forward_events(as_of: date, days: int = 2) -> List[Dict[str, Any]]:
    from sqlalchemy import text
    from src.db.session import engine

    end = as_of + timedelta(days=days)
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT symbol, event_type, event_time
            FROM event.events
            WHERE event_time::date >= :start AND event_time::date <= :end
            ORDER BY event_time ASC
            LIMIT 20
        """), {"start": as_of, "end": end})
        rows = result.fetchall()

    return [
        {
            "date": r.event_time.strftime("%Y-%m-%d %H:%M") if hasattr(r.event_time, "strftime") else str(r.event_time),
            "event": r.event_type,
            "symbol": r.symbol,
        }
        for r in rows
    ]


async def _fetch_last_debate_summary() -> Optional[Dict[str, Any]]:
    from sqlalchemy import text
    from src.db.session import engine

    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT payload, run_time FROM gpt.recommendations
            WHERE provider = 'debate'
            ORDER BY run_time DESC LIMIT 1
        """))
        row = result.fetchone()

    if not row:
        return None

    payload = row.payload
    if isinstance(payload, str):
        payload = json.loads(payload)

    synthesis = payload.get("synthesis", {})
    return {
        "run_time": row.run_time.isoformat() if hasattr(row.run_time, "isoformat") else str(row.run_time),
        "regime": synthesis.get("regime_consensus", ""),
        "consensus_trades": synthesis.get("consensus_trades", []),
        "risk_warnings": synthesis.get("risk_warnings", []),
        "portfolio_actions": synthesis.get("portfolio_actions", []),
    }


# ── Feed fetching ────────────────────────────────────────────────────────

def _fetch_feeds() -> List[Dict[str, Any]]:
    """Fetch tech/social feeds (synchronous — runs in thread)."""
    from src.feeds.fetcher import fetch_all
    try:
        items = fetch_all()
        logger.info(f"Fetched {len(items)} feed items")
        return items
    except Exception as e:
        logger.warning(f"Feed fetch failed: {e}")
        return []


# ── LLM commentary (conversational Mira-style) ──────────────────────────

async def _generate_market_commentary(
    llm_client,
    context: Dict[str, Any],
    today: date,
) -> str:
    """Generate conversational market narrative — Mira tone."""
    prompt = f"""今天是 {today.isoformat()}，美股开盘前。你在给老板写晨间 briefing。

以下是你拿到的数据：
{json.dumps(context, indent=2, default=str)}

## 你的任务

写一份 **综合晨间分析**，像跟一个很懂市场的朋友聊天。不要用"本报告分析了"这种腔调，直接说重点。

结构：
1. **隔夜发生了什么** — 全球市场、重要新闻、overnight futures 动向。哪些是噪音哪些是信号，说清楚。
2. **今天要看什么** — 催化剂、数据发布、earnings、关键事件。标出具体时间。
3. **持仓检查** — 现有仓位（META、GOOGL 等）在当前环境下怎么看？有没有需要注意的。
4. **关键水位** — 对持仓和大盘指数，今天哪些价位重要？

## 语气

- 像微信语音转文字，不是 sell-side research
- 有态度：觉得市场在扯淡就说扯淡，看好就说看好
- 用具体数字，不要泛泛而谈
- 中文为主，术语可以用英文（VIX、put/call ratio、gamma 等）
- 控制在 500-800 字"""

    system = (
        "你是一个资深量化 PM 的晨间分析助理。你的风格：直接、有态度、数据驱动。"
        "像跟同事聊天一样写，不是写报告。用中文，术语保持英文。"
        "不要客套，不要免责声明。每句话都要有信息量。"
    )

    try:
        return await llm_client.generate(prompt, system=system, temperature=0.4)
    except Exception as e:
        logger.warning(f"Market commentary failed: {e}")
        return f"*晨间分析暂时不可用: {e}*"


async def _generate_feed_narrative(
    llm_client,
    feed_items: List[Dict[str, Any]],
    today: date,
) -> str:
    """Generate conversational tech/social feed narrative — Mira tone."""
    if not feed_items:
        return ""

    # Compact feed items for prompt
    compact = []
    for item in feed_items[:50]:
        compact.append({
            "source": item.get("source", ""),
            "title": item.get("title", ""),
            "summary": item.get("summary", "")[:200],
            "url": item.get("url", ""),
            "score": item.get("score", 0),
        })

    prompt = f"""今天是 {today.isoformat()}。以下是从 Arxiv、Hacker News、Reddit、GitHub Trending、RSS 等抓到的最新内容：

{json.dumps(compact, indent=2, ensure_ascii=False)}

## 你的任务

挑 5-7 个最有意思/最重要的，用聊天的方式讲核心想法。

规则：
- 不要用"本文探讨了"、"该研究提出"这种论文腔
- 用自己的话解释，不是翻译摘要
- 有态度：觉得牛就说牛，觉得扯就说扯
- 链接用 markdown 嵌在话里面，不要单独列出来
- 条与条之间自然过渡，不要机械列举
- 如果有什么跟市场/投资有关的（比如 AI 突破对 META/GOOGL 的影响），特别点出来
- 最后一句你的真实感想
- 控制在 400-600 字"""

    system = (
        "你是一个既懂技术又懂市场的朋友，在跟人聊今天看到了什么好玩的。"
        "你有自己的判断力，不是新闻播报员。中文为主，术语英文。"
    )

    try:
        return await llm_client.generate(prompt, system=system, temperature=0.5)
    except Exception as e:
        logger.warning(f"Feed narrative failed: {e}")
        return ""


async def _generate_cross_signal(
    llm_client,
    market_commentary: str,
    feed_narrative: str,
    portfolio_symbols: List[str],
    today: date,
) -> str:
    """Find cross-signal insights between market data and tech/social feeds."""
    if not feed_narrative:
        return ""

    prompt = f"""今天是 {today.isoformat()}。

以下是今天的市场分析：
---
{market_commentary}
---

以下是今天的技术/社会动态：
---
{feed_narrative}
---

当前持仓：{', '.join(portfolio_symbols)}

## 你的任务

找出市场信号和技术/社会信号之间的 **交叉洞察**。比如：
- 某个 AI 突破 → 对 META/GOOGL 意味着什么？
- 监管新闻 → 对科技股的影响？
- 开源趋势 → 对商业 AI 公司的威胁/机会？
- 宏观事件 → 对科技行业资金流向的影响？

只写真正有交叉价值的洞察，不要硬凑。如果确实没有交叉信号，就说"今天市场和技术面没有明显交叉"然后简短收尾。

控制在 200-400 字。"""

    system = (
        "你是一个同时看市场和技术的分析师，擅长发现不同领域之间的关联。"
        "只说有实际意义的交叉洞察，不要硬凑。中文，直接。"
    )

    try:
        return await llm_client.generate(prompt, system=system, temperature=0.4)
    except Exception as e:
        logger.warning(f"Cross-signal analysis failed: {e}")
        return ""


# ── PDF generation ────────────────────────────────────────────────────────

async def _generate_premarket_pdf(
    sections: Dict[str, Any],
    today: date,
) -> str:
    """Generate morning briefing PDF."""
    from src.report.delivery import generate_pdf, TEMPLATE_DIR, DEFAULT_OUTPUT_DIR

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(DEFAULT_OUTPUT_DIR / f"premarket_{today.isoformat()}.pdf")

    pdf_path = await generate_pdf(
        sections=sections,
        output_path=output_path,
        template_name="premarket_report.html",
    )
    logger.info(f"Morning PDF: {pdf_path}")
    return pdf_path


# ── Briefing assembly ────────────────────────────────────────────────────

def _assemble_briefing_md(
    today: date,
    commentary: str,
    feed_narrative: str,
    cross_signal: str,
    news: List[Dict[str, Any]],
    polymarket: List[Dict[str, Any]],
    portfolio: Dict[str, Any],
    rates: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    last_debate: Optional[Dict[str, Any]],
) -> str:
    """Assemble the comprehensive morning briefing as markdown (for Mira app)."""
    lines = [f"# Tetra Morning Briefing — {today.isoformat()}\n"]
    lines.append(f"*Generated {datetime.now(tz=UTC).strftime('%H:%M UTC')} · Before US Market Open*\n")

    # Portfolio snapshot
    summary = portfolio.get("summary")
    if summary:
        lines.append(
            f"**Portfolio:** ${summary['total_value']:,.0f}"
            f" | Last day: {summary['daily_return']:+.2%}"
            f" | Cumulative: {summary['cumulative_return']:+.2%}\n"
        )

    if last_debate and last_debate.get("regime"):
        lines.append(f"**Regime:** {last_debate['regime']}\n")

    # Market commentary
    if commentary:
        lines.append(f"\n---\n\n{commentary}\n")

    # Feed narrative
    if feed_narrative:
        lines.append(f"\n---\n\n## Tech & Social\n\n{feed_narrative}\n")

    # Cross-signal
    if cross_signal:
        lines.append(f"\n---\n\n## Cross-Signal Insights\n\n{cross_signal}\n")

    # Rates
    if rates:
        lines.append("\n## Rates & Macro\n")
        for r in rates:
            lines.append(f"- **{r['name']}:** {r['value']:.2f}")

    # Top overnight news
    if news:
        lines.append("\n## Overnight News (Top 15)\n")
        for a in news[:15]:
            sentiment_tag = ""
            sent = a.get("sentiment") or 0
            if sent > 0.3:
                sentiment_tag = " [+]"
            elif sent < -0.3:
                sentiment_tag = " [-]"
            syms = a.get("symbols") or []
            symbols_str = f" ({', '.join(syms[:3])})" if syms else ""
            title = a.get("title") or "(no headline)"
            lines.append(f"- {title}{symbols_str}{sentiment_tag}")

    # Polymarket
    if polymarket:
        top_poly = [p for p in polymarket if p.get("volume_24h", 0) > 0][:8]
        if top_poly:
            lines.append("\n## Polymarket Top Markets\n")
            for p in top_poly:
                lines.append(f"- **{p['question']}** → {p['price']:.0%}")

    # Events
    if events:
        lines.append("\n## Today's Calendar\n")
        for e in events:
            lines.append(f"- {e['date']} | **{e['symbol']}** {e['event']}")

    # Portfolio positions
    positions = portfolio.get("positions", [])
    if positions:
        lines.append("\n## Portfolio Positions\n")
        lines.append("| Symbol | Price | Value | Weight | P&L |")
        lines.append("|--------|------:|------:|-------:|----:|")
        for p in positions:
            lines.append(
                f"| {p['symbol']} | ${p['price']:,.2f} | ${p['value']:,.0f} | {p['weight']:.1f}% | ${p['pnl']:+,.0f} |"
            )

    # Risk warnings
    if last_debate and last_debate.get("risk_warnings"):
        lines.append("\n## Risk Warnings\n")
        for w in last_debate["risk_warnings"]:
            lines.append(f"- ⚠ {w}")

    return "\n".join(lines)


# ── Delivery ─────────────────────────────────────────────────────────────

def _push_to_mira(briefing_md: str, today: date) -> Dict[str, Any]:
    """Push morning briefing to Mira bridge + artifacts."""
    import uuid
    from src.mira.push import BRIDGE_OUTBOX, BRIEFINGS_DIR

    results: Dict[str, Any] = {}

    # Bridge message (short summary for notification)
    try:
        BRIDGE_OUTBOX.mkdir(parents=True, exist_ok=True)
        msg_id = uuid.uuid4().hex[:8]
        ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

        summary_lines = [l for l in briefing_md.split("\n") if l.strip() and not l.startswith("#") and not l.startswith("*")]
        short_summary = summary_lines[0][:200] if summary_lines else "Morning briefing ready"

        message = {
            "id": msg_id,
            "sender": "agent",
            "timestamp": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": "text",
            "content": f"☀️ Morning Briefing {today}\n\n{short_summary}",
            "thread_id": "",
        }

        msg_path = BRIDGE_OUTBOX / f"agent_{ts}_{msg_id}.json"
        msg_path.write_text(json.dumps(message, indent=2), encoding="utf-8")
        results["bridge_message"] = str(msg_path)
        logger.info(f"Bridge message: {msg_path.name}")
    except Exception as e:
        logger.warning(f"Bridge push failed: {e}")
        results["bridge_error"] = str(e)

    # Briefing artifact (full MD for Mira app)
    try:
        BRIEFINGS_DIR.mkdir(parents=True, exist_ok=True)
        briefing_path = BRIEFINGS_DIR / f"{today.isoformat()}_premarket.md"
        briefing_path.write_text(briefing_md, encoding="utf-8")
        results["briefing"] = str(briefing_path)
        logger.info(f"Briefing artifact: {briefing_path.name}")
    except Exception as e:
        logger.warning(f"Briefing write failed: {e}")
        results["briefing_error"] = str(e)

    return results


async def _send_pdf_email(
    pdf_path: str,
    today: date,
) -> bool:
    """Send morning briefing PDF via email."""
    from src.report.delivery import send_email

    if not settings.email_enabled:
        logger.info("Email disabled in settings")
        return False

    recipients = [r.strip() for r in settings.email_recipients.split(",") if r.strip()]
    if not recipients or not settings.smtp_username or not settings.smtp_password:
        logger.warning("Email skipped: missing recipients or SMTP credentials")
        return False

    subject = f"Tetra Morning Briefing — {today.isoformat()}"
    return await send_email(
        pdf_path=pdf_path,
        recipients=recipients,
        smtp_username=settings.smtp_username,
        smtp_password=settings.smtp_password,
        subject=subject,
    )


# ── Main ──────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> None:
    today = date.today()
    logger.info(f"Generating comprehensive morning briefing for {today}")

    # Step 1: Data ingest
    if not args.skip_ingest:
        logger.info("Running pre-market data ingest...")
        try:
            ingest_summary = await _ingest_premarket_data()
            logger.info(f"Ingest complete: {ingest_summary}")
        except Exception as e:
            logger.warning(f"Ingest failed (continuing with stale data): {e}")

    # Step 1.5: Mark-to-market portfolio with latest prices
    logger.info("Marking portfolio to market...")
    try:
        from src.portfolio.manager import update_positions
        mtm_result = await update_positions(today)
        logger.info(f"Portfolio MTM: {mtm_result.get('status', 'unknown')}")
    except Exception as e:
        logger.warning(f"Portfolio MTM failed (using stale data): {e}")

    # Step 2: Fetch market data from DB + feeds in parallel
    logger.info("Fetching data...")

    # DB fetches (async)
    news_task = asyncio.create_task(_fetch_overnight_news())
    narrative_task = asyncio.create_task(_fetch_narrative_state(today))
    polymarket_task = asyncio.create_task(_fetch_polymarket_signals())
    portfolio_task = asyncio.create_task(_fetch_portfolio_snapshot())
    rates_task = asyncio.create_task(_fetch_rates_snapshot())
    events_task = asyncio.create_task(_fetch_forward_events(today))
    debate_task = asyncio.create_task(_fetch_last_debate_summary())

    # Feeds (sync in thread)
    feed_items = []
    if not args.skip_feeds:
        loop = asyncio.get_event_loop()
        feed_items = await loop.run_in_executor(None, _fetch_feeds)

    # Await all DB fetches
    news = await news_task
    narrative = await narrative_task
    polymarket = await polymarket_task
    portfolio = await portfolio_task
    rates = await rates_task
    events = await events_task
    last_debate = await debate_task

    logger.info(
        f"Data: {len(news)} news, {len(polymarket)} polymarket, "
        f"{len(portfolio.get('positions', []))} positions, "
        f"{len(rates)} rates, {len(events)} events, {len(feed_items)} feeds"
    )

    # Step 3: LLM commentary (three passes)
    commentary = ""
    feed_narrative = ""
    cross_signal = ""

    if not args.no_llm:
        llm_client = get_llm_client(args.llm_provider)
        if llm_client:
            # Pass 1 & 2: Market commentary + feed narrative in parallel
            logger.info("Generating LLM narratives...")
            commentary_context = {
                "date": today.isoformat(),
                "overnight_news": news[:20],
                "narrative": narrative,
                "polymarket_movers": [p for p in polymarket if abs(p.get("change_24h", 0)) > 0.02],
                "portfolio": portfolio.get("summary"),
                "positions": [p["symbol"] for p in portfolio.get("positions", [])],
                "rates": rates,
                "events": events,
                "last_regime": last_debate.get("regime") if last_debate else None,
                "last_risk_warnings": last_debate.get("risk_warnings", []) if last_debate else [],
                "last_consensus_trades": last_debate.get("consensus_trades", []) if last_debate else [],
            }

            market_task = asyncio.create_task(
                _generate_market_commentary(llm_client, commentary_context, today)
            )
            feed_task = asyncio.create_task(
                _generate_feed_narrative(llm_client, feed_items, today)
            ) if feed_items else None

            commentary = await market_task
            if feed_task:
                feed_narrative = await feed_task

            # Pass 3: Cross-signal synthesis (depends on passes 1 & 2)
            if feed_narrative:
                logger.info("Generating cross-signal synthesis...")
                portfolio_symbols = [p["symbol"] for p in portfolio.get("positions", [])]
                cross_signal = await _generate_cross_signal(
                    llm_client, commentary, feed_narrative, portfolio_symbols, today
                )
        else:
            logger.warning("No LLM client available, skipping commentary")

    # Step 4: Assemble markdown briefing
    briefing_md = _assemble_briefing_md(
        today=today,
        commentary=commentary,
        feed_narrative=feed_narrative,
        cross_signal=cross_signal,
        news=news,
        polymarket=polymarket,
        portfolio=portfolio,
        rates=rates,
        events=events,
        last_debate=last_debate,
    )

    # Step 5: Save markdown locally
    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"premarket_{today.isoformat()}.md"
    md_path.write_text(briefing_md, encoding="utf-8")
    logger.info(f"Saved MD: {md_path}")

    # Step 6: Generate PDF
    pdf_path = None
    if not args.no_pdf:
        try:
            pdf_sections = {
                "commentary": commentary,
                "feed_narrative": feed_narrative,
                "cross_signal": cross_signal,
                "news": news,
                "polymarket": polymarket,
                "portfolio": portfolio,
                "rates": rates,
                "events": events,
                "narrative": narrative,
                "regime": last_debate.get("regime") if last_debate else None,
                "risk_warnings": last_debate.get("risk_warnings", []) if last_debate else [],
            }
            pdf_path = await _generate_premarket_pdf(pdf_sections, today)
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")

    # Step 7: Email PDF
    if pdf_path and not args.no_email:
        sent = await _send_pdf_email(pdf_path, today)
        logger.info(f"Email: {'sent' if sent else 'failed'}")

    # Step 8: Push to Mira
    mira_result = _push_to_mira(briefing_md, today)
    logger.info(f"Mira push: {mira_result}")

    print(f"\nMorning briefing generated:")
    print(f"  MD: {md_path}")
    if pdf_path:
        print(f"  PDF: {pdf_path}")
    print(f"  Mira: {mira_result}")


def main() -> None:
    setup_logging()
    args = parse_args()

    # Skip on weekends
    today = date.today()
    if today.weekday() >= 5:
        logger.info(f"Skipping morning briefing on {today.strftime('%A')}")
        return

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
