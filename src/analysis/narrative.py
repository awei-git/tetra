"""Narrative Fragmentation Index + LLM Narrative Shift Detection.

Core insight (Bertsch, Hull & Zhang): narratives CONSOLIDATE during expansions
and FRAGMENT during contractions. We measure the ENTROPY of the narrative
distribution — not whether sentiment is positive or negative, but HOW MANY
competing explanations exist simultaneously.

Pipeline:
1. Fetch recent news articles from DB
2. Compute narrative fragmentation (Shannon entropy of topic distribution)
3. Detect topic distribution shifts vs prior window
4. LLM: classify whether detected shifts are genuine regime changes
5. Store results in narrative.daily_state
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from src.db.session import engine

logger = logging.getLogger(__name__)
UTC = timezone.utc

# Narrative topics — broader than the existing 8 macro topics.
# LLM will extract these from article text; we also use keyword fallback.
NARRATIVE_THEMES = [
    "ai_growth", "ai_regulation", "rate_cuts", "rate_hikes",
    "recession_fear", "soft_landing", "earnings_momentum", "earnings_miss",
    "geopolitical_risk", "trade_war", "fiscal_stimulus", "inflation_persistent",
    "inflation_cooling", "labor_strong", "labor_weak", "credit_stress",
    "crypto_rally", "energy_shock", "consumer_strong", "consumer_weak",
    "tech_antitrust", "supply_chain", "housing_crisis", "dollar_strong",
    "dollar_weak", "china_slowdown", "europe_risk", "election",
    "fed_hawkish", "fed_dovish", "other",
]

# Keyword mapping for fast first-pass classification (no LLM needed)
THEME_KEYWORDS: Dict[str, List[str]] = {
    "ai_growth": ["artificial intelligence", "ai chip", "gpu demand", "ai infrastructure", "generative ai", "ai capex"],
    "ai_regulation": ["ai regulation", "ai safety", "ai ban", "ai oversight"],
    "rate_cuts": ["rate cut", "lower rates", "dovish fed", "easing cycle", "fed pivot"],
    "rate_hikes": ["rate hike", "higher rates", "hawkish fed", "tightening"],
    "recession_fear": ["recession", "economic downturn", "contraction", "hard landing"],
    "soft_landing": ["soft landing", "goldilocks", "no recession"],
    "earnings_momentum": ["earnings beat", "revenue beat", "guidance raise", "strong earnings"],
    "earnings_miss": ["earnings miss", "revenue miss", "guidance cut", "weak earnings", "profit warning"],
    "geopolitical_risk": ["war", "military", "sanctions", "conflict", "invasion", "missile"],
    "trade_war": ["tariff", "trade war", "import duty", "trade restriction", "trade ban"],
    "fiscal_stimulus": ["stimulus", "government spending", "infrastructure bill", "fiscal"],
    "inflation_persistent": ["sticky inflation", "inflation higher", "cpi above", "prices rising"],
    "inflation_cooling": ["inflation cooling", "cpi below", "disinflation", "prices falling"],
    "labor_strong": ["jobs added", "low unemployment", "hiring surge", "labor tight"],
    "labor_weak": ["layoffs", "job cuts", "unemployment rising", "hiring freeze"],
    "credit_stress": ["credit spread", "default", "bank failure", "liquidity crisis", "credit crunch"],
    "crypto_rally": ["bitcoin rally", "crypto surge", "btc all-time", "crypto etf"],
    "energy_shock": ["oil spike", "energy crisis", "opec cut", "gas prices"],
    "consumer_strong": ["consumer spending", "retail sales beat", "consumer confidence up"],
    "consumer_weak": ["consumer pullback", "retail sales miss", "consumer confidence down"],
    "tech_antitrust": ["antitrust", "monopoly", "breakup", "regulation tech"],
    "supply_chain": ["supply chain", "shipping disruption", "shortage"],
    "housing_crisis": ["housing crash", "mortgage crisis", "home prices fall"],
    "dollar_strong": ["dollar strength", "dxy rally", "strong dollar"],
    "dollar_weak": ["dollar weakness", "dxy fall", "weak dollar"],
    "china_slowdown": ["china slowdown", "china deflation", "china property"],
    "europe_risk": ["eurozone", "europe recession", "ecb"],
    "election": ["election", "presidential", "midterm", "polling"],
    "fed_hawkish": ["fed hawkish", "higher for longer", "no cut"],
    "fed_dovish": ["fed dovish", "rate path lower", "accommodative"],
}


async def _fetch_articles(as_of: date, lookback_days: int = 7) -> List[Dict[str, Any]]:
    """Fetch recent news articles with title and content."""
    query = text("""
        SELECT id, headline, COALESCE(summary, '') AS summary,
               source, published_at, sentiment,
               COALESCE(tickers, ARRAY[]::varchar[]) AS tickers
        FROM news.articles
        WHERE published_at >= CAST(:start AS TIMESTAMPTZ)
          AND published_at <= CAST(:end AS TIMESTAMPTZ) + INTERVAL '1 day'
        ORDER BY published_at DESC
    """)
    start = as_of - timedelta(days=lookback_days)
    async with engine.begin() as conn:
        result = await conn.execute(query, {"start": start, "end": as_of})
        rows = result.fetchall()
    return [
        {
            "id": r.id,
            "title": r.headline,
            "content": r.summary,
            "source": r.source,
            "published_at": r.published_at,
            "sentiment": float(r.sentiment) if r.sentiment is not None else None,
            "tickers": list(r.tickers),
        }
        for r in rows
    ]


def _classify_themes_keyword(articles: List[Dict[str, Any]]) -> List[Dict[str, List[str]]]:
    """Fast keyword-based theme classification. Returns themes per article."""
    results = []
    for article in articles:
        text_lower = f"{article['title']} {article['content'][:2000]}".lower()
        themes = []
        for theme, keywords in THEME_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                themes.append(theme)
        if not themes:
            themes = ["other"]
        results.append({"id": article["id"], "themes": themes})
    return results


def _compute_entropy(theme_counts: Counter) -> float:
    """Shannon entropy of theme distribution. Higher = more fragmented."""
    total = sum(theme_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in theme_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def _compute_max_entropy(n_themes: int) -> float:
    """Maximum possible entropy for n themes (uniform distribution)."""
    if n_themes <= 1:
        return 0.0
    return math.log2(n_themes)


def _compute_fragmentation_index(theme_counts: Counter) -> float:
    """Normalized fragmentation: 0 = single narrative, 1 = maximum fragmentation."""
    n = len(theme_counts)
    max_ent = _compute_max_entropy(n)
    if max_ent == 0:
        return 0.0
    return _compute_entropy(theme_counts) / max_ent


def _compute_distribution_shift(
    current_counts: Counter, prior_counts: Counter
) -> Tuple[float, Dict[str, float]]:
    """Jensen-Shannon divergence between two theme distributions.

    Returns (divergence, per_theme_delta).
    JSD is symmetric and bounded [0, 1] when using log2.
    """
    all_themes = set(current_counts.keys()) | set(prior_counts.keys())
    if not all_themes:
        return 0.0, {}

    total_curr = max(sum(current_counts.values()), 1)
    total_prior = max(sum(prior_counts.values()), 1)

    p = {t: current_counts.get(t, 0) / total_curr for t in all_themes}
    q = {t: prior_counts.get(t, 0) / total_prior for t in all_themes}

    # M = (P + Q) / 2
    m = {t: (p[t] + q[t]) / 2 for t in all_themes}

    def kl(dist, ref):
        total = 0.0
        for t in all_themes:
            if dist[t] > 0 and ref[t] > 0:
                total += dist[t] * math.log2(dist[t] / ref[t])
        return total

    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)

    # Per-theme change
    delta = {}
    for t in all_themes:
        d = p.get(t, 0) - q.get(t, 0)
        if abs(d) > 0.02:  # Only report meaningful shifts
            delta[t] = round(d, 4)

    return jsd, delta


async def _llm_narrative_analysis(
    llm_client,
    theme_counts: Counter,
    fragmentation: float,
    jsd: float,
    theme_deltas: Dict[str, float],
    sample_headlines: List[str],
    as_of: date,
) -> Dict[str, Any]:
    """Use LLM to interpret the quantitative signals and extract narrative state."""
    # Sort themes by count
    top_themes = theme_counts.most_common(10)
    rising = {k: v for k, v in sorted(theme_deltas.items(), key=lambda x: -x[1]) if v > 0}
    falling = {k: v for k, v in sorted(theme_deltas.items(), key=lambda x: x[1]) if v < 0}

    prompt = f"""You are a market narrative analyst. Analyze the current narrative landscape.

Date: {as_of.isoformat()}

QUANTITATIVE SIGNALS:
- Narrative Fragmentation Index: {fragmentation:.3f} (0=single dominant narrative, 1=maximum fragmentation)
  Higher fragmentation historically correlates with regime transitions and increased volatility.
- Distribution Shift (JSD): {jsd:.4f} (0=no change, 1=complete change vs prior week)
- Top themes by frequency: {json.dumps(top_themes)}
- Rising themes (vs prior week): {json.dumps(rising)}
- Falling themes (vs prior week): {json.dumps(falling)}

SAMPLE HEADLINES (most recent 30):
{chr(10).join(f'- {h}' for h in sample_headlines[:30])}

RESPOND WITH JSON ONLY:
{{
  "dominant_narrative": "one sentence describing THE dominant market story right now",
  "narrative_shift": float between -1 and 1 (negative = shifting bearish, positive = shifting bullish, 0 = no shift),
  "shift_magnitude": float 0 to 1 (how significant is the current shift),
  "counter_narrative": "the strongest counter-narrative that could disrupt the dominant one",
  "novelty": float 0 to 1 (how NEW is the current narrative mix vs typical patterns),
  "regime_signal": "risk_on" | "risk_off" | "transition" | "uncertain",
  "expected_impact": {{
    "direction": "bullish" | "bearish" | "neutral",
    "magnitude": "large" | "moderate" | "small",
    "timeframe": "days" | "weeks" | "months",
    "reasoning": "one sentence"
  }},
  "historical_parallel": "brief description of the most similar past narrative environment"
}}"""

    system = "You are a quantitative market narrative analyst. Respond only with valid JSON. No markdown."

    try:
        raw = await llm_client.generate(prompt, system=system, temperature=0.2)
        # Extract JSON from response
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"LLM narrative analysis failed: {e}")
        return {
            "dominant_narrative": f"Top theme: {top_themes[0][0] if top_themes else 'unknown'}",
            "narrative_shift": 0.0,
            "shift_magnitude": jsd,
            "counter_narrative": None,
            "novelty": fragmentation,
            "regime_signal": "uncertain",
            "expected_impact": None,
            "historical_parallel": None,
        }


async def _store_narrative_state(
    as_of: date,
    fragmentation: float,
    jsd: float,
    theme_counts: Counter,
    theme_deltas: Dict[str, float],
    llm_analysis: Dict[str, Any],
) -> None:
    """Store narrative state in narrative.daily_state."""
    query = text("""
        INSERT INTO narrative.daily_state
          (date, scope, scope_value, dominant_narrative, narrative_shift,
           shift_magnitude, counter_narrative, novelty,
           historical_parallels, expected_impact, raw_analysis)
        VALUES
          (:date, 'market', 'market',
           :dominant_narrative, :narrative_shift,
           :shift_magnitude, :counter_narrative, :novelty,
           CAST(:historical_parallels AS JSONB),
           CAST(:expected_impact AS JSONB),
           :raw_analysis)
        ON CONFLICT (date, scope, scope_value)
        DO UPDATE SET
          dominant_narrative = EXCLUDED.dominant_narrative,
          narrative_shift = EXCLUDED.narrative_shift,
          shift_magnitude = EXCLUDED.shift_magnitude,
          counter_narrative = EXCLUDED.counter_narrative,
          novelty = EXCLUDED.novelty,
          historical_parallels = EXCLUDED.historical_parallels,
          expected_impact = EXCLUDED.expected_impact,
          raw_analysis = EXCLUDED.raw_analysis
    """)
    raw_analysis = json.dumps({
        "fragmentation_index": fragmentation,
        "jsd": jsd,
        "theme_counts": dict(theme_counts),
        "theme_deltas": theme_deltas,
        "llm": llm_analysis,
    })
    async with engine.begin() as conn:
        await conn.execute(query, {
            "date": as_of,
            "dominant_narrative": llm_analysis.get("dominant_narrative", ""),
            "narrative_shift": llm_analysis.get("narrative_shift", 0.0),
            "shift_magnitude": llm_analysis.get("shift_magnitude", jsd),
            "counter_narrative": llm_analysis.get("counter_narrative"),
            "novelty": llm_analysis.get("novelty", fragmentation),
            "historical_parallels": json.dumps(
                llm_analysis.get("historical_parallel")
            ),
            "expected_impact": json.dumps(
                llm_analysis.get("expected_impact")
            ),
            "raw_analysis": raw_analysis,
        })


async def run_narrative_analysis(
    as_of: Optional[date] = None,
    llm_client=None,
) -> Dict[str, Any]:
    """Run the full narrative fragmentation pipeline.

    1. Fetch articles for current and prior windows
    2. Classify themes (keyword-based, fast)
    3. Compute fragmentation index + distribution shift
    4. LLM enrichment (if client provided)
    5. Store results

    Returns summary dict.
    """
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    logger.info(f"Running narrative analysis for {as_of}")

    # Current window: last 7 days
    current_articles = await _fetch_articles(as_of, lookback_days=7)
    # Prior window: 7-14 days ago
    prior_articles = await _fetch_articles(
        as_of - timedelta(days=7), lookback_days=7
    )

    if not current_articles:
        logger.warning("No articles found for narrative analysis")
        return {"as_of": as_of.isoformat(), "status": "no_data"}

    # Classify themes
    current_classified = _classify_themes_keyword(current_articles)
    prior_classified = _classify_themes_keyword(prior_articles)

    # Count themes
    current_counts: Counter = Counter()
    for item in current_classified:
        for theme in item["themes"]:
            current_counts[theme] += 1

    prior_counts: Counter = Counter()
    for item in prior_classified:
        for theme in item["themes"]:
            prior_counts[theme] += 1

    # Compute metrics
    fragmentation = _compute_fragmentation_index(current_counts)
    jsd, theme_deltas = _compute_distribution_shift(current_counts, prior_counts)

    sample_headlines = [a["title"] for a in current_articles if a["title"]]

    # LLM enrichment
    llm_analysis = {}
    if llm_client:
        llm_analysis = await _llm_narrative_analysis(
            llm_client, current_counts, fragmentation, jsd,
            theme_deltas, sample_headlines, as_of,
        )
    else:
        # Fallback: quantitative-only state
        top_theme = current_counts.most_common(1)
        llm_analysis = {
            "dominant_narrative": top_theme[0][0] if top_theme else "unknown",
            "narrative_shift": 0.0,
            "shift_magnitude": jsd,
            "counter_narrative": None,
            "novelty": fragmentation,
            "regime_signal": "uncertain",
            "expected_impact": None,
            "historical_parallel": None,
        }

    # Store
    await _store_narrative_state(
        as_of, fragmentation, jsd, current_counts, theme_deltas, llm_analysis,
    )

    result = {
        "as_of": as_of.isoformat(),
        "status": "success",
        "articles_current": len(current_articles),
        "articles_prior": len(prior_articles),
        "fragmentation_index": round(fragmentation, 4),
        "distribution_shift_jsd": round(jsd, 4),
        "top_themes": current_counts.most_common(5),
        "rising_themes": {k: v for k, v in theme_deltas.items() if v > 0},
        "falling_themes": {k: v for k, v in theme_deltas.items() if v < 0},
        "regime_signal": llm_analysis.get("regime_signal", "uncertain"),
        "dominant_narrative": llm_analysis.get("dominant_narrative", ""),
    }
    logger.info(
        f"Narrative analysis complete: fragmentation={fragmentation:.3f}, "
        f"shift={jsd:.4f}, regime={result['regime_signal']}"
    )
    return result
