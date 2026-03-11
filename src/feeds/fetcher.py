"""Fetch content from web sources for Tetra morning briefings.

Ported from Mira explorer — fetches arxiv, Reddit, HuggingFace, GitHub,
Hacker News, Lobsters, Dev.to, and RSS feeds. Used in pre-market briefing
to provide tech/social context alongside market data.
"""

import json
import logging
import urllib.request
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

USER_AGENT = "TetraAgent/1.0 (market research)"

SOURCES_FILE = Path(__file__).resolve().parents[2] / "config" / "sources.json"
FEEDS_DIR = Path(__file__).resolve().parents[2] / "feeds"
MAX_FEED_ITEMS = 100


def load_sources() -> dict:
    if not SOURCES_FILE.exists():
        logger.warning("No sources.json found at %s", SOURCES_FILE)
        return {}
    try:
        return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load sources.json: %s", e)
        return {}


def _http_get(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ── Arxiv ──────────────────────────────────────────────────────────────────

def fetch_arxiv(categories: list[str], max_results: int = 10) -> list[dict]:
    if not categories:
        return []
    cat_query = "+OR+".join(f"cat:{c}" for c in categories)
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={cat_query}"
        f"&sortBy=submittedDate&sortOrder=descending"
        f"&max_results={max_results}"
    )
    try:
        xml_text = _http_get(url, timeout=20)
    except Exception as e:
        logger.error("Arxiv fetch failed: %s", e)
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items = []
    try:
        root = ET.fromstring(xml_text)
        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
            summary = entry.findtext("atom:summary", "", ns).strip()[:300]
            link = ""
            for lnk in entry.findall("atom:link", ns):
                if lnk.get("type") == "text/html":
                    link = lnk.get("href", "")
                    break
            if not link:
                link_el = entry.find("atom:id", ns)
                link = link_el.text if link_el is not None else ""
            items.append({"source": "arxiv", "title": title, "summary": summary, "url": link})
    except ET.ParseError as e:
        logger.error("Arxiv XML parse failed: %s", e)
    return items


# ── Reddit ─────────────────────────────────────────────────────────────────

def fetch_reddit(subreddits: list[str], limit: int = 10) -> list[dict]:
    items = []
    for sub in subreddits:
        url = f"https://old.reddit.com/r/{sub}/hot.json?limit={limit}"
        try:
            data = json.loads(_http_get(url))
        except Exception as e:
            logger.error("Reddit fetch failed for r/%s: %s", sub, e)
            continue
        for post in data.get("data", {}).get("children", []):
            d = post.get("data", {})
            if d.get("stickied"):
                continue
            items.append({
                "source": f"r/{sub}",
                "title": d.get("title", ""),
                "summary": (d.get("selftext", "") or "")[:300],
                "url": f"https://reddit.com{d.get('permalink', '')}",
                "score": d.get("score", 0),
            })
    return items


# ── HuggingFace Daily Papers ──────────────────────────────────────────────

def fetch_hf_papers() -> list[dict]:
    url = "https://huggingface.co/api/daily_papers"
    try:
        data = json.loads(_http_get(url))
    except Exception as e:
        logger.error("HuggingFace fetch failed: %s", e)
        return []
    items = []
    for paper in data[:15]:
        p = paper.get("paper", {})
        items.append({
            "source": "huggingface",
            "title": p.get("title", ""),
            "summary": (p.get("summary", "") or "")[:300],
            "url": f"https://huggingface.co/papers/{p.get('id', '')}",
        })
    return items


# ── GitHub Trending ────────────────────────────────────────────────────────

def fetch_github_trending(days_back: int = 7, language: str | None = None,
                          per_page: int = 25) -> list[dict]:
    since = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    q = f"created:>{since}"
    if language:
        q += f" language:{language}"
    url = (
        f"https://api.github.com/search/repositories"
        f"?q={urllib.parse.quote(q)}&sort=stars&order=desc&per_page={per_page}"
    )
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.github+json",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        logger.error("GitHub trending fetch failed: %s", e)
        return []
    items = []
    for r in data.get("items", []):
        desc = (r.get("description") or "")[:300]
        lang = r.get("language") or "?"
        stars = r.get("stargazers_count", 0)
        items.append({
            "source": "github_trending",
            "title": f"{r['full_name']} [{lang}] ({stars} stars)",
            "summary": desc,
            "url": r["html_url"],
            "stars": stars,
        })
    return items


# ── Hacker News ────────────────────────────────────────────────────────────

def fetch_hackernews(count: int = 30, min_points: int = 0) -> list[dict]:
    params = f"tags=front_page&hitsPerPage={count}"
    if min_points:
        params += f"&numericFilters=points>{min_points}"
    url = f"https://hn.algolia.com/api/v1/search?{params}"
    try:
        data = json.loads(_http_get(url, timeout=15))
    except Exception as e:
        logger.error("HackerNews fetch failed: %s", e)
        return []
    items = []
    for h in data.get("hits", []):
        items.append({
            "source": "hackernews",
            "title": h.get("title", ""),
            "summary": f"Score: {h.get('points', 0)} | Comments: {h.get('num_comments', 0)}",
            "url": h.get("url") or f"https://news.ycombinator.com/item?id={h['objectID']}",
            "score": h.get("points", 0),
        })
    return items


# ── Lobsters ───────────────────────────────────────────────────────────────

def fetch_lobsters(count: int = 25) -> list[dict]:
    url = "https://lobste.rs/hottest.json"
    try:
        data = json.loads(_http_get(url, timeout=15))
    except Exception as e:
        logger.error("Lobsters fetch failed: %s", e)
        return []
    items = []
    for s in data[:count]:
        tags = ", ".join(s.get("tags", []))
        items.append({
            "source": "lobsters",
            "title": s.get("title", ""),
            "summary": f"[{tags}] Score: {s.get('score', 0)}",
            "url": s.get("url") or s.get("short_id_url", ""),
            "score": s.get("score", 0),
        })
    return items


# ── RSS feeds ──────────────────────────────────────────────────────────────

def fetch_rss(feeds: list[dict]) -> list[dict]:
    items = []
    for feed in feeds:
        name = feed.get("name", "RSS")
        url = feed.get("url", "")
        if not url:
            continue
        try:
            xml_text = _http_get(url, timeout=15)
        except Exception as e:
            logger.error("RSS fetch failed for '%s': %s", name, e)
            continue
        try:
            root = ET.fromstring(xml_text)
            feed_items = []
            for item in root.findall(".//item")[:10]:
                feed_items.append({
                    "source": name,
                    "title": (item.findtext("title") or "").strip(),
                    "summary": (item.findtext("description") or "").strip()[:300],
                    "url": (item.findtext("link") or "").strip(),
                })
            if not feed_items:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns)[:10]:
                    link = ""
                    for lnk in entry.findall("atom:link", ns):
                        link = lnk.get("href", "")
                        break
                    feed_items.append({
                        "source": name,
                        "title": (entry.findtext("atom:title", "", ns) or "").strip(),
                        "summary": (entry.findtext("atom:summary", "", ns) or "").strip()[:300],
                        "url": link,
                    })
            items.extend(feed_items)
        except ET.ParseError as e:
            logger.error("RSS parse failed for '%s': %s", name, e)
    return items


# ── Fetch all ──────────────────────────────────────────────────────────────

def fetch_sources(source_names: list[str]) -> list[dict]:
    """Fetch from specific named sources. Returns combined list of items."""
    sources = load_sources()
    if not sources:
        return []

    all_items: list[dict] = []
    names_lower = [n.strip().lower() for n in source_names]

    if "arxiv" in names_lower:
        cfg = sources.get("arxiv", {})
        if cfg.get("categories"):
            items = fetch_arxiv(cfg["categories"], cfg.get("max_results", 10))
            all_items.extend(items)
            logger.info("Arxiv: %d items", len(items))

    if "reddit" in names_lower:
        cfg = sources.get("reddit", {})
        if cfg.get("subreddits"):
            items = fetch_reddit(cfg["subreddits"], cfg.get("limit", 10))
            all_items.extend(items)
            logger.info("Reddit: %d items", len(items))

    if "huggingface" in names_lower:
        if sources.get("huggingface", {}).get("enabled", True):
            items = fetch_hf_papers()
            all_items.extend(items)
            logger.info("HuggingFace: %d items", len(items))

    if "github" in names_lower or "github_trending" in names_lower:
        cfg = sources.get("github_trending", {})
        if cfg.get("enabled", True):
            items = fetch_github_trending(
                days_back=cfg.get("days_back", 7),
                language=cfg.get("language"),
                per_page=cfg.get("per_page", 25),
            )
            all_items.extend(items)
            logger.info("GitHub Trending: %d items", len(items))

    if "hackernews" in names_lower or "hacker_news" in names_lower:
        cfg = sources.get("hackernews", {})
        if cfg.get("enabled", True):
            items = fetch_hackernews(
                count=cfg.get("count", 30),
                min_points=cfg.get("min_points", 50),
            )
            all_items.extend(items)
            logger.info("HackerNews: %d items", len(items))

    if "lobsters" in names_lower:
        cfg = sources.get("lobsters", {})
        if cfg.get("enabled", True):
            items = fetch_lobsters(count=cfg.get("count", 25))
            all_items.extend(items)
            logger.info("Lobsters: %d items", len(items))

    rss_feeds = sources.get("rss", [])
    if "rss" in names_lower:
        if rss_feeds:
            items = fetch_rss(rss_feeds)
            all_items.extend(items)
            logger.info("RSS (all): %d items", len(items))

    logger.info("Feed fetch (%s): %d items total", ",".join(source_names), len(all_items))

    # Save raw
    raw_path = FEEDS_DIR / "raw" / f"{datetime.now().strftime('%Y-%m-%d_%H%M')}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(all_items, indent=2, ensure_ascii=False), encoding="utf-8")

    return all_items[:MAX_FEED_ITEMS]


def fetch_all() -> list[dict]:
    """Fetch from all configured sources."""
    return fetch_sources([
        "arxiv", "reddit", "huggingface", "github_trending",
        "hackernews", "lobsters", "rss",
    ])


def fetch_market_relevant() -> list[dict]:
    """Fetch sources most relevant for morning market briefing.

    Focuses on HN, Reddit (economics, markets), and market-related RSS.
    Skips pure-tech sources like Lobsters/Dev.to.
    """
    return fetch_sources([
        "hackernews", "reddit", "rss",
    ])
