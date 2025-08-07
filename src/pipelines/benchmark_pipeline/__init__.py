"""Benchmark pipeline for strategy backtesting and performance tracking."""

from .pipeline import BenchmarkPipeline
from .main import main

__all__ = ["BenchmarkPipeline", "main"]