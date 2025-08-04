"""Main signal computer that orchestrates all signal computations."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime, timedelta
import warnings
import logging

from .types import SignalResult, SignalMetadata, SignalType
from .config import SignalConfig
from .base_signal import BaseSignal


logger = logging.getLogger(__name__)


class SignalComputer:
    """Orchestrates computation of all signals."""
    
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.config.validate()
        
        # Signal registry
        self._signals: Dict[str, BaseSignal] = {}
        self._signal_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize thread pool for parallel computation
        if self.config.parallel_compute:
            self._executor = ThreadPoolExecutor(max_workers=4)
        else:
            self._executor = None
    
    def register_signal(self, signal: BaseSignal) -> None:
        """Register a signal computer."""
        if signal.name in self._signals:
            logger.warning(f"Signal {signal.name} already registered, overwriting")
        self._signals[signal.name] = signal
    
    def register_signals(self, signals: List[BaseSignal]) -> None:
        """Register multiple signals."""
        for signal in signals:
            self.register_signal(signal)
    
    def compute(self, 
                data: pd.DataFrame,
                symbols: Optional[Union[str, List[str]]] = None,
                signal_names: Optional[List[str]] = None,
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None) -> SignalResult:
        """Compute signals for given data.
        
        Args:
            data: DataFrame with OHLCV data, must have 'symbol' column for multi-symbol
            symbols: Symbol(s) to compute signals for
            signal_names: Specific signals to compute (None = all)
            start_date: Start date for computation
            end_date: End date for computation
            
        Returns:
            SignalResult containing all computed signals
        """
        start_time = time.time()
        
        # Filter data
        filtered_data = self._filter_data(data, symbols, start_date, end_date)
        
        # Determine which signals to compute
        signals_to_compute = self._get_signals_to_compute(signal_names)
        
        # Check if we have single or multi-symbol data
        is_multi_symbol = 'symbol' in filtered_data.columns
        
        if is_multi_symbol:
            result = self._compute_multi_symbol(filtered_data, signals_to_compute)
        else:
            result = self._compute_single_symbol(filtered_data, signals_to_compute)
        
        # Set compute time
        result.compute_time = time.time() - start_time
        
        return result
    
    def _filter_data(self, 
                     data: pd.DataFrame,
                     symbols: Optional[Union[str, List[str]]],
                     start_date: Optional[datetime],
                     end_date: Optional[datetime]) -> pd.DataFrame:
        """Filter data by symbols and date range."""
        filtered = data.copy()
        
        # Filter by symbols
        if symbols is not None:
            if isinstance(symbols, str):
                symbols = [symbols]
            if 'symbol' in filtered.columns:
                filtered = filtered[filtered['symbol'].isin(symbols)]
        
        # Filter by date range
        if 'date' in filtered.columns:
            if start_date:
                filtered = filtered[filtered['date'] >= start_date]
            if end_date:
                filtered = filtered[filtered['date'] <= end_date]
        elif filtered.index.name == 'date' or isinstance(filtered.index, pd.DatetimeIndex):
            if start_date:
                filtered = filtered[filtered.index >= start_date]
            if end_date:
                filtered = filtered[filtered.index <= end_date]
        
        return filtered
    
    def _get_signals_to_compute(self, signal_names: Optional[List[str]]) -> List[str]:
        """Determine which signals to compute based on config and request."""
        if signal_names:
            # Validate requested signals exist
            invalid = [s for s in signal_names if s not in self._signals]
            if invalid:
                raise ValueError(f"Unknown signals: {invalid}")
            return signal_names
        
        # Get all signals based on config
        signals = []
        
        if self.config.compute_technical:
            tech_signals = [s for s, sig in self._signals.items() 
                          if sig.signal_type in [SignalType.TREND, SignalType.MOMENTUM, 
                                                SignalType.VOLATILITY, SignalType.VOLUME, 
                                                SignalType.PATTERN]]
            if self.config.technical_signals:
                tech_signals = [s for s in tech_signals if s in self.config.technical_signals]
            signals.extend(tech_signals)
        
        if self.config.compute_statistical:
            stat_signals = [s for s, sig in self._signals.items()
                          if sig.signal_type in [SignalType.STATISTICAL, SignalType.CORRELATION,
                                                SignalType.DISTRIBUTION, SignalType.REGIME]]
            if self.config.statistical_signals:
                stat_signals = [s for s in stat_signals if s in self.config.statistical_signals]
            signals.extend(stat_signals)
        
        if self.config.compute_ml:
            ml_signals = [s for s, sig in self._signals.items()
                        if sig.signal_type in [SignalType.ML_CLASSIFICATION, SignalType.ML_REGRESSION,
                                              SignalType.ML_CLUSTERING, SignalType.ML_ANOMALY]]
            if self.config.ml_signals:
                ml_signals = [s for s in ml_signals if s in self.config.ml_signals]
            signals.extend(ml_signals)
        
        return signals
    
    def _compute_single_symbol(self, data: pd.DataFrame, signal_names: List[str]) -> SignalResult:
        """Compute signals for single symbol data."""
        result_data = pd.DataFrame(index=data.index)
        metadata = {}
        errors = {}
        warnings = []
        
        # Check data length
        if len(data) < self.config.min_data_points:
            warnings.append(f"Data has only {len(data)} points, minimum is {self.config.min_data_points}")
        
        # Compute signals
        if self.config.parallel_compute and self._executor and len(signal_names) > 1:
            # Parallel computation
            futures = {}
            for signal_name in signal_names:
                if signal_name in self._signals:
                    future = self._executor.submit(
                        self._compute_signal_safe,
                        self._signals[signal_name],
                        data
                    )
                    futures[future] = signal_name
            
            # Collect results
            for future in as_completed(futures):
                signal_name = futures[future]
                try:
                    signal_data, signal_meta = future.result()
                    if signal_data is not None:
                        result_data[signal_name] = signal_data
                        metadata[signal_name] = signal_meta
                except Exception as e:
                    errors[signal_name] = str(e)
                    logger.error(f"Error computing {signal_name}: {e}")
        else:
            # Sequential computation
            for signal_name in signal_names:
                if signal_name in self._signals:
                    try:
                        signal_data, signal_meta = self._compute_signal_safe(
                            self._signals[signal_name], data
                        )
                        if signal_data is not None:
                            result_data[signal_name] = signal_data
                            metadata[signal_name] = signal_meta
                    except Exception as e:
                        errors[signal_name] = str(e)
                        logger.error(f"Error computing {signal_name}: {e}")
        
        return SignalResult(
            data=result_data,
            metadata=metadata,
            compute_time=0,  # Will be set by caller
            errors=errors,
            warnings=warnings
        )
    
    def _compute_multi_symbol(self, data: pd.DataFrame, signal_names: List[str]) -> SignalResult:
        """Compute signals for multi-symbol data."""
        # Group by symbol and compute
        all_results = []
        all_metadata = {}
        all_errors = {}
        all_warnings = []
        
        for symbol, symbol_data in data.groupby('symbol'):
            # Remove symbol column for computation
            symbol_data = symbol_data.drop(columns=['symbol'])
            
            # Compute signals for this symbol
            result = self._compute_single_symbol(symbol_data, signal_names)
            
            # Add symbol prefix to column names
            result.data.columns = [f"{symbol}_{col}" for col in result.data.columns]
            
            # Update metadata with symbol prefix
            for signal_name, meta in result.metadata.items():
                all_metadata[f"{symbol}_{signal_name}"] = meta
            
            # Update errors with symbol prefix
            for signal_name, error in result.errors.items():
                all_errors[f"{symbol}_{signal_name}"] = error
            
            # Add warnings with symbol prefix
            for warning in result.warnings:
                all_warnings.append(f"{symbol}: {warning}")
            
            all_results.append(result.data)
        
        # Combine all results
        combined_data = pd.concat(all_results, axis=1)
        
        return SignalResult(
            data=combined_data,
            metadata=all_metadata,
            compute_time=0,  # Will be set by caller
            errors=all_errors,
            warnings=all_warnings
        )
    
    def _compute_signal_safe(self, signal: BaseSignal, data: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[SignalMetadata]]:
        """Safely compute a single signal with error handling."""
        try:
            # Check cache if enabled
            if self.config.cache_results:
                cache_key = self._get_cache_key(signal.name, data)
                if cache_key in self._signal_cache:
                    cache_entry = self._signal_cache[cache_key]
                    if (datetime.now() - cache_entry['timestamp']).total_seconds() < self.config.cache_ttl_seconds:
                        return cache_entry['data'], cache_entry['metadata']
            
            # Compute signal
            signal_data, metadata = signal.compute_with_metadata(data)
            
            # Cache result if enabled
            if self.config.cache_results:
                self._signal_cache[cache_key] = {
                    'data': signal_data,
                    'metadata': metadata,
                    'timestamp': datetime.now()
                }
            
            return signal_data, metadata
            
        except Exception as e:
            logger.error(f"Error in signal {signal.name}: {str(e)}")
            raise
    
    def _get_cache_key(self, signal_name: str, data: pd.DataFrame) -> str:
        """Generate cache key for signal and data."""
        # Simple cache key based on signal name and data shape/dates
        if isinstance(data.index, pd.DatetimeIndex):
            date_range = f"{data.index[0]}_{data.index[-1]}"
        else:
            date_range = f"{len(data)}"
        
        return f"{signal_name}_{date_range}_{data.shape}"
    
    def list_signals(self) -> List[Dict[str, Any]]:
        """List all registered signals."""
        return [
            {
                'name': signal.name,
                'type': signal.signal_type.value,
                'description': signal.description,
                'dependencies': signal.dependencies
            }
            for signal in self._signals.values()
        ]
    
    def get_signal_info(self, signal_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a signal."""
        if signal_name not in self._signals:
            return None
        
        signal = self._signals[signal_name]
        return {
            'name': signal.name,
            'type': signal.signal_type.value,
            'description': signal.description,
            'dependencies': signal.dependencies,
            'parameters': signal.get_parameters()
        }
    
    def clear_cache(self) -> None:
        """Clear signal cache."""
        self._signal_cache.clear()
    
    def __del__(self):
        """Cleanup executor on deletion."""
        if self._executor:
            self._executor.shutdown(wait=False)