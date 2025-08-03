"""Lazy evaluation system for signal computation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Set
from functools import lru_cache
import weakref
import logging

from ..base import BaseSignal, SignalResult
from ..base.types import SignalMetadata

logger = logging.getLogger(__name__)


class LazySignalEvaluator:
    """Evaluator for lazy signal computation with dependency resolution."""
    
    def __init__(self):
        self._signals: Dict[str, BaseSignal] = {}
        self._data_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._computation_graph: Dict[str, Set[str]] = {}
        self._computed_signals: Dict[str, pd.Series] = {}
        self._metadata: Dict[str, SignalMetadata] = {}
    
    def register_signal(self, signal: BaseSignal) -> None:
        """Register a signal and its dependencies."""
        self._signals[signal.name] = signal
        
        # Build dependency graph
        self._computation_graph[signal.name] = set(signal.dependencies)
    
    def register_signals(self, signals: List[BaseSignal]) -> None:
        """Register multiple signals."""
        for signal in signals:
            self.register_signal(signal)
    
    def create_lazy_dataframe(self, data: pd.DataFrame) -> 'LazySignalDataFrame':
        """Create a lazy DataFrame that computes signals on demand."""
        return LazySignalDataFrame(data, self)
    
    def compute_signal(self, 
                      signal_name: str,
                      data: pd.DataFrame,
                      computed_signals: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """Compute a single signal, resolving dependencies as needed.
        
        Args:
            signal_name: Name of signal to compute
            data: Input OHLCV data
            computed_signals: Already computed signals
            
        Returns:
            Computed signal series
        """
        if computed_signals is None:
            computed_signals = {}
        
        # Check if already computed
        if signal_name in computed_signals:
            return computed_signals[signal_name]
        
        # Get signal
        if signal_name not in self._signals:
            raise ValueError(f"Unknown signal: {signal_name}")
        
        signal = self._signals[signal_name]
        
        # Resolve dependencies first
        dependencies_data = {}
        for dep in signal.dependencies:
            if dep in ['open', 'high', 'low', 'close', 'volume']:
                # Base data column
                if dep in data.columns:
                    dependencies_data[dep] = data[dep]
                else:
                    raise ValueError(f"Required column {dep} not in data")
            else:
                # Another signal
                dep_signal = self.compute_signal(dep, data, computed_signals)
                dependencies_data[dep] = dep_signal
                computed_signals[dep] = dep_signal
        
        # Prepare input data
        if dependencies_data:
            input_data = pd.DataFrame(dependencies_data)
        else:
            input_data = data
        
        # Compute signal
        logger.debug(f"Computing signal: {signal_name}")
        result, metadata = signal.compute_with_metadata(input_data)
        
        # Cache results
        computed_signals[signal_name] = result
        self._metadata[signal_name] = metadata
        
        return result
    
    def get_computation_order(self, signal_names: List[str]) -> List[str]:
        """Get the order in which signals should be computed based on dependencies.
        
        Uses topological sort to determine computation order.
        """
        # Build graph of only requested signals and their dependencies
        relevant_graph = {}
        to_visit = set(signal_names)
        visited = set()
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            
            visited.add(current)
            
            if current in self._computation_graph:
                deps = self._computation_graph[current]
                relevant_graph[current] = deps
                to_visit.update(deps - visited)
        
        # Topological sort
        result = []
        in_degree = {node: 0 for node in relevant_graph}
        
        # Calculate in-degrees
        for node, deps in relevant_graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Find nodes with no dependencies
        queue = [node for node, degree in in_degree.items() if degree == 0]
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Update in-degrees
            for node, deps in relevant_graph.items():
                if current in deps:
                    in_degree[node] -= 1
                    if in_degree[node] == 0:
                        queue.append(node)
        
        # Filter to only requested signals
        return [s for s in result if s in signal_names]
    
    def compute_batch(self,
                     signal_names: List[str],
                     data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute multiple signals efficiently by resolving shared dependencies.
        
        Args:
            signal_names: List of signals to compute
            data: Input data
            
        Returns:
            Dictionary of computed signals
        """
        # Get optimal computation order
        computation_order = self.get_computation_order(signal_names)
        
        computed_signals = {}
        
        # Compute in order
        for signal_name in computation_order:
            self.compute_signal(signal_name, data, computed_signals)
        
        # Return only requested signals
        return {name: computed_signals[name] for name in signal_names}


class LazySignalDataFrame:
    """DataFrame-like object that computes signals lazily."""
    
    def __init__(self, data: pd.DataFrame, evaluator: LazySignalEvaluator):
        self._base_data = data
        self._evaluator = evaluator
        self._computed_signals: Dict[str, pd.Series] = {}
        
        # Cache for getitem access
        self._getitem_cache = {}
    
    @property
    def index(self):
        """Return the index of the base data."""
        return self._base_data.index
    
    @property
    def columns(self):
        """Return available columns (base + signals)."""
        base_cols = list(self._base_data.columns)
        signal_cols = list(self._evaluator._signals.keys())
        return base_cols + signal_cols
    
    def __getitem__(self, key):
        """Access columns or signals lazily."""
        if isinstance(key, str):
            # Single column/signal access
            if key in self._base_data.columns:
                return self._base_data[key]
            elif key in self._evaluator._signals:
                if key not in self._computed_signals:
                    self._computed_signals[key] = self._evaluator.compute_signal(
                        key, self._base_data, self._computed_signals
                    )
                return self._computed_signals[key]
            else:
                raise KeyError(f"Column/signal '{key}' not found")
        
        elif isinstance(key, list):
            # Multiple column access
            result_data = {}
            
            for col in key:
                if col in self._base_data.columns:
                    result_data[col] = self._base_data[col]
                elif col in self._evaluator._signals:
                    if col not in self._computed_signals:
                        self._computed_signals[col] = self._evaluator.compute_signal(
                            col, self._base_data, self._computed_signals
                        )
                    result_data[col] = self._computed_signals[col]
                else:
                    raise KeyError(f"Column/signal '{col}' not found")
            
            return pd.DataFrame(result_data)
        
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
    def compute(self, signal_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Force computation of specified signals or all signals.
        
        Args:
            signal_names: List of signals to compute (None = all)
            
        Returns:
            DataFrame with base data and computed signals
        """
        if signal_names is None:
            signal_names = list(self._evaluator._signals.keys())
        
        # Compute signals
        computed = self._evaluator.compute_batch(signal_names, self._base_data)
        self._computed_signals.update(computed)
        
        # Combine with base data
        result = self._base_data.copy()
        for name, series in computed.items():
            result[name] = series
        
        return result
    
    def get_dependencies(self, signal_name: str) -> Set[str]:
        """Get dependencies for a signal."""
        if signal_name not in self._evaluator._signals:
            return set()
        
        dependencies = set()
        to_visit = [signal_name]
        
        while to_visit:
            current = to_visit.pop()
            if current in self._evaluator._computation_graph:
                deps = self._evaluator._computation_graph[current]
                dependencies.update(deps)
                to_visit.extend([d for d in deps if d in self._evaluator._signals])
        
        return dependencies
    
    def explain_computation(self, signal_name: str) -> str:
        """Explain how a signal will be computed."""
        if signal_name not in self._evaluator._signals:
            return f"Signal '{signal_name}' not found"
        
        signal = self._evaluator._signals[signal_name]
        deps = self.get_dependencies(signal_name)
        
        explanation = f"Signal: {signal_name}\n"
        explanation += f"Description: {signal.description}\n"
        explanation += f"Direct dependencies: {signal.dependencies}\n"
        explanation += f"All dependencies: {deps}\n"
        
        # Get computation order
        all_signals = list(deps) + [signal_name]
        all_signals = [s for s in all_signals if s in self._evaluator._signals]
        order = self._evaluator.get_computation_order(all_signals)
        
        explanation += f"Computation order: {' -> '.join(order)}"
        
        return explanation
    
    def memory_estimate(self, signal_names: Optional[List[str]] = None) -> Dict[str, int]:
        """Estimate memory usage for computing signals."""
        if signal_names is None:
            signal_names = list(self._evaluator._signals.keys())
        
        estimates = {}
        base_size = self._base_data.memory_usage(deep=True).sum()
        estimates['base_data'] = base_size
        
        # Estimate per signal (simplified)
        rows = len(self._base_data)
        float_size = np.dtype(np.float64).itemsize
        
        for signal_name in signal_names:
            # Assume each signal produces one float64 column
            estimates[signal_name] = rows * float_size
        
        estimates['total'] = base_size + sum(
            v for k, v in estimates.items() if k != 'base_data'
        )
        
        return estimates