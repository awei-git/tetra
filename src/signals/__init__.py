"""
Comprehensive signal computation module for quantitative trading.

This module provides a wide range of technical, statistical, and ML-based signals
for trading strategy development.
"""

from .base import (
    SignalComputer,
    SignalConfig,
    SignalResult,
    SignalType,
    SignalMetadata
)

from .technical import TechnicalSignals
from .statistical import StatisticalSignals
from .ml import MLSignals

__all__ = [
    'SignalComputer',
    'SignalConfig',
    'SignalResult',
    'SignalType',
    'SignalMetadata',
    'TechnicalSignals',
    'StatisticalSignals',
    'MLSignals',
]

__version__ = '1.0.0'