"""Data providers for ingestion module."""

from .base import BaseProvider
from .polygon import PolygonProvider
from .yfinance import YFinanceProvider
from .alphavantage import AlphaVantageProvider
from .fred import FREDProvider
from .newsapi import NewsAPIProvider

__all__ = [
    'BaseProvider',
    'PolygonProvider',
    'YFinanceProvider',
    'AlphaVantageProvider',
    'FREDProvider',
    'NewsAPIProvider'
]