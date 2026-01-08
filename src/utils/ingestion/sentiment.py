"""Sentiment analysis helpers for news ingestion."""

from __future__ import annotations

from typing import Optional, Tuple

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - optional dependency
    SentimentIntensityAnalyzer = None


_ANALYZER = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None


def analyze_sentiment(text: str) -> Tuple[Optional[float], Optional[float], str]:
    """Return (score, confidence, label) for a text snippet."""
    if not text:
        return None, None, "neutral"
    if _ANALYZER is None:
        return None, None, "neutral"

    scores = _ANALYZER.polarity_scores(text)
    score = float(scores.get("compound", 0.0))
    confidence = abs(score)
    if score >= 0.05:
        label = "positive"
    elif score <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return score, confidence, label


def normalize_sentiment(score: Optional[float]) -> Tuple[Optional[float], Optional[float], str]:
    if score is None:
        return None, None, "neutral"
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None, None, "neutral"
    confidence = abs(score)
    if score >= 0.05:
        label = "positive"
    elif score <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return score, confidence, label
