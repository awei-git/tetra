"""
Market simulator framework for backtesting trading strategies.

This module provides a comprehensive framework for:
- Historical market simulation
- Portfolio tracking and management
- Performance metrics calculation
- Event-based market replay
"""

from .base import BaseSimulator, SimulationConfig, SimulationResult
from .historical import HistoricalSimulator, EventPeriod
from .portfolio import Portfolio, Position, Transaction

__all__ = [
    "BaseSimulator",
    "SimulationConfig", 
    "SimulationResult",
    "HistoricalSimulator",
    "EventPeriod",
    "Portfolio",
    "Position",
    "Transaction",
]