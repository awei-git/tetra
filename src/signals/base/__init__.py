"""Base classes and interfaces for signal computation."""

from .signal_computer import SignalComputer
from .config import SignalConfig
from .types import SignalResult, SignalType, SignalMetadata
from .base_signal import BaseSignal

__all__ = [
    'SignalComputer',
    'SignalConfig',
    'SignalResult',
    'SignalType',
    'SignalMetadata',
    'BaseSignal',
]