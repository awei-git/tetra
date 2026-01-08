"""Topic tagging helpers for macro/news classification."""

from __future__ import annotations

from typing import Dict, List

_TOPICS: Dict[str, List[str]] = {
    "inflation": ["inflation", "cpi", "ppi", "pce", "price pressures"],
    "rates": ["fomc", "fed", "interest rate", "rates", "yield", "treasury", "bond"],
    "growth": ["gdp", "growth", "recession", "slowdown"],
    "labor": ["jobs", "employment", "unemployment", "payroll"],
    "housing": ["housing", "mortgage", "home sales"],
    "energy": ["oil", "crude", "gas", "energy"],
    "fx": ["dollar", "usd", "yen", "euro", "fx"],
    "geopolitics": ["war", "sanction", "tariff", "geopolitical"],
}


def detect_macro_topics(text: str) -> List[str]:
    if not text:
        return []
    lowered = text.lower()
    matches = []
    for topic, keywords in _TOPICS.items():
        for keyword in keywords:
            if keyword in lowered:
                matches.append(topic)
                break
    return sorted(set(matches))
