"""Factor scoring utilities for alpha ranking."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.utils.factors.definitions import get_factor_definitions

ACTION_THRESHOLD = 0.2
MIN_SAMPLES = 5


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _stddev(values: List[float]) -> float:
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / max(1, len(values) - 1)
    return math.sqrt(variance)


def _normalize_signal(value: float, direction: float, scale: float = 2.0) -> float:
    score = math.tanh((value * direction) / scale)
    return max(-1.0, min(1.0, score))


def action_from_signal(signal: Optional[float], threshold: float = ACTION_THRESHOLD) -> str:
    if signal is None:
        return "neutral"
    if signal >= threshold:
        return "buy"
    if signal <= -threshold:
        return "sell"
    return "neutral"


def build_factor_stats(
    rows: Iterable[Tuple[str, str, float]],
    definitions: Dict[str, Dict[str, object]],
    min_samples: int = MIN_SAMPLES,
) -> Dict[str, Dict[str, float]]:
    factor_values: Dict[str, List[float]] = {}
    for symbol, factor, value in rows:
        if factor not in definitions:
            continue
        if symbol == "__macro__":
            continue
        definition = definitions[factor]
        if definition.get("normalization") in ("self", "count"):
            continue
        factor_values.setdefault(factor, []).append(value)

    stats: Dict[str, Dict[str, float]] = {}
    for factor, values in factor_values.items():
        if len(values) < min_samples:
            continue
        std = _stddev(values)
        if std == 0:
            continue
        stats[factor] = {"mean": _mean(values), "std": std, "count": float(len(values))}
    return stats


def compute_factor_signal(
    factor: str,
    value: float,
    definition: Dict[str, object],
    stats: Dict[str, Dict[str, float]],
) -> Optional[float]:
    if value is None:
        return None
    direction = float(definition.get("direction", 1.0))
    normalization = definition.get("normalization", "zscore")
    if normalization == "self":
        scale = float(definition.get("scale", 2.0))
        return _normalize_signal(float(value), direction, scale=scale)
    if normalization == "count":
        if float(value) <= 0:
            return 0.0
        window = float(definition.get("window", 1) or 1)
        scale = float(definition.get("scale", max(5.0, window)))
        return _normalize_signal(float(value), direction, scale=max(scale, 1.0))
    if normalization == "delta":
        if abs(float(value)) < 1e-9:
            return 0.0
        window = float(definition.get("window", 1) or 1)
        scale = float(definition.get("scale", max(5.0, window)))
        return _normalize_signal(float(value), direction, scale=max(scale, 1.0))
    stat = stats.get(factor)
    if not stat:
        return None
    zscore = (float(value) - stat["mean"]) / stat["std"]
    return _normalize_signal(zscore, direction)


def score_symbol_values(
    symbol_values: Dict[str, float],
    definitions: Dict[str, Dict[str, object]],
    stats: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    total_weight = 0.0
    score = 0.0
    contributions: List[Tuple[str, float]] = []
    used = 0

    for factor, value in symbol_values.items():
        definition = definitions.get(factor)
        if not definition:
            continue
        signal = compute_factor_signal(factor, value, definition, stats)
        if signal is None:
            continue
        weight = abs(float(definition.get("weight", 0.05)))
        if weight <= 0:
            continue
        contribution = signal * weight
        total_weight += weight
        score += contribution
        used += 1
        contributions.append((factor, contribution))

    if used == 0 or total_weight == 0:
        return {"score": None, "coverage": 0, "contributions": []}

    contributions.sort(key=lambda item: abs(item[1]), reverse=True)
    return {
        "score": score / total_weight,
        "coverage": used,
        "contributions": contributions,
    }


def score_factor_rows(rows: Iterable[Tuple[str, str, float]]) -> Dict[str, Any]:
    definitions = get_factor_definitions()
    factor_stats = build_factor_stats(rows, definitions)
    symbol_values: Dict[str, Dict[str, float]] = {}
    for symbol, factor, value in rows:
        if factor not in definitions:
            continue
        symbol_values.setdefault(symbol, {})[factor] = value

    scores: Dict[str, Dict[str, Any]] = {}
    for symbol, values in symbol_values.items():
        scores[symbol] = score_symbol_values(values, definitions, factor_stats)

    return {
        "scores": scores,
        "factors_used": list(factor_stats.keys()),
        "definitions": definitions,
    }
