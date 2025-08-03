"""Memory-optimized signal computation strategies."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Iterator, Tuple
import gc
import logging

from ..base import SignalComputer, SignalConfig, SignalResult
from ..base.types import SignalMetadata

logger = logging.getLogger(__name__)


class MemoryOptimizedComputer(SignalComputer):
    """Signal computer with memory optimization strategies."""
    
    def __init__(self, 
                 config: Optional[SignalConfig] = None,
                 chunk_size: int = 10000,
                 dtype_optimization: bool = True):
        super().__init__(config)
        self.chunk_size = chunk_size
        self.dtype_optimization = dtype_optimization
    
    def compute_chunked(self,
                       data: pd.DataFrame,
                       signal_names: Optional[List[str]] = None,
                       chunk_size: Optional[int] = None) -> SignalResult:
        """Compute signals in chunks to reduce memory usage.
        
        Useful for very large datasets that don't fit in memory.
        """
        chunk_size = chunk_size or self.chunk_size
        
        # Determine signals to compute
        signals_to_compute = self._get_signals_to_compute(signal_names)
        
        # Initialize result containers
        chunk_results = []
        all_metadata = {}
        all_errors = {}
        all_warnings = []
        
        # Process data in chunks
        total_rows = len(data)
        n_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        logger.info(f"Processing {total_rows} rows in {n_chunks} chunks")
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)
            
            # Extract chunk with overlap for indicators that need history
            overlap = min(200, start_idx)  # Look back up to 200 periods
            chunk_start = max(0, start_idx - overlap)
            
            chunk_data = data.iloc[chunk_start:end_idx].copy()
            
            # Optimize data types if enabled
            if self.dtype_optimization:
                chunk_data = self._optimize_dtypes(chunk_data)
            
            # Compute signals for this chunk
            chunk_result = super().compute(chunk_data, signal_names=signals_to_compute)
            
            # Keep only the non-overlapping part
            if overlap > 0 and i > 0:
                chunk_result.data = chunk_result.data.iloc[overlap:]
            
            chunk_results.append(chunk_result.data)
            
            # Update metadata and errors (only from first chunk)
            if i == 0:
                all_metadata.update(chunk_result.metadata)
                all_errors.update(chunk_result.errors)
            
            all_warnings.extend(chunk_result.warnings)
            
            # Force garbage collection after each chunk
            del chunk_data
            del chunk_result
            gc.collect()
        
        # Combine all chunks
        combined_data = pd.concat(chunk_results, axis=0, ignore_index=True)
        
        return SignalResult(
            data=combined_data,
            metadata=all_metadata,
            compute_time=0,  # Will be set by caller
            errors=all_errors,
            warnings=all_warnings
        )
    
    def compute_streaming(self,
                         data_iterator: Iterator[pd.DataFrame],
                         signal_names: Optional[List[str]] = None,
                         lookback: int = 200) -> Iterator[SignalResult]:
        """Compute signals on streaming data.
        
        Args:
            data_iterator: Iterator yielding DataFrames
            signal_names: Signals to compute
            lookback: Number of periods to keep for indicators
            
        Yields:
            SignalResult for each chunk
        """
        # Determine signals to compute
        signals_to_compute = self._get_signals_to_compute(signal_names)
        
        # Buffer for maintaining lookback window
        buffer = pd.DataFrame()
        
        for chunk in data_iterator:
            # Optimize data types
            if self.dtype_optimization:
                chunk = self._optimize_dtypes(chunk)
            
            # Combine with buffer
            if not buffer.empty:
                combined = pd.concat([buffer, chunk], axis=0, ignore_index=True)
            else:
                combined = chunk
            
            # Compute signals
            result = super().compute(combined, signal_names=signals_to_compute)
            
            # Extract only the new results (not from buffer)
            if not buffer.empty:
                new_results = result.data.iloc[len(buffer):]
                result.data = new_results
            
            # Update buffer (keep last lookback periods)
            if len(combined) > lookback:
                buffer = combined.iloc[-lookback:].copy()
            else:
                buffer = combined.copy()
            
            yield result
            
            # Clean up
            gc.collect()
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types to reduce memory usage."""
        optimized = df.copy()
        
        for col in optimized.columns:
            col_type = optimized[col].dtype
            
            # Optimize numeric columns
            if col_type != 'object':
                c_min = optimized[col].min()
                c_max = optimized[col].max()
                
                # Integer optimization
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized[col] = optimized[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized[col] = optimized[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized[col] = optimized[col].astype(np.int32)
                
                # Float optimization
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        optimized[col] = optimized[col].astype(np.float32)
            
            # Optimize object columns (try to convert to category)
            else:
                num_unique = optimized[col].nunique()
                num_total = len(optimized[col])
                if num_unique / num_total < 0.5:  # Less than 50% unique
                    optimized[col] = optimized[col].astype('category')
        
        # Log memory reduction
        orig_size = df.memory_usage(deep=True).sum()
        new_size = optimized.memory_usage(deep=True).sum()
        reduction_pct = (1 - new_size / orig_size) * 100
        
        logger.debug(f"Memory optimized: {orig_size:,} -> {new_size:,} bytes "
                    f"({reduction_pct:.1f}% reduction)")
        
        return optimized
    
    def compute_lazy(self,
                    data: pd.DataFrame,
                    signal_names: Optional[List[str]] = None) -> 'LazySignalResult':
        """Return a lazy evaluation wrapper for signals.
        
        Signals are only computed when accessed.
        """
        return LazySignalResult(self, data, signal_names)
    
    def estimate_memory_usage(self,
                            data: pd.DataFrame,
                            signal_names: Optional[List[str]] = None) -> Dict[str, int]:
        """Estimate memory usage for computing signals.
        
        Returns:
            Dictionary with memory estimates in bytes
        """
        signals_to_compute = self._get_signals_to_compute(signal_names)
        
        estimates = {
            'input_data': data.memory_usage(deep=True).sum(),
            'signals': {}
        }
        
        # Estimate per signal (rough approximation)
        rows = len(data)
        float_size = np.dtype(np.float64).itemsize
        
        for signal_name in signals_to_compute:
            signal = self._signals.get(signal_name)
            if signal:
                # Base estimate: one float64 column
                base_estimate = rows * float_size
                
                # Adjust based on signal type
                if hasattr(signal, 'output_columns'):
                    n_outputs = len(signal.output_columns)
                else:
                    n_outputs = 1
                
                estimates['signals'][signal_name] = base_estimate * n_outputs
        
        estimates['total_signals'] = sum(estimates['signals'].values())
        estimates['total'] = estimates['input_data'] + estimates['total_signals']
        
        return estimates


class LazySignalResult:
    """Lazy evaluation wrapper for signal results."""
    
    def __init__(self, 
                 computer: MemoryOptimizedComputer,
                 data: pd.DataFrame,
                 signal_names: Optional[List[str]] = None):
        self.computer = computer
        self.data = data
        self.signal_names = signal_names
        self._cache = {}
        self._metadata = {}
    
    def __getitem__(self, signal_name: str) -> pd.Series:
        """Compute and return a specific signal."""
        if signal_name not in self._cache:
            # Compute just this signal
            result = self.computer.compute(
                self.data,
                signal_names=[signal_name]
            )
            
            if signal_name in result.data.columns:
                self._cache[signal_name] = result.data[signal_name]
                self._metadata[signal_name] = result.metadata.get(signal_name, {})
            else:
                raise KeyError(f"Signal {signal_name} not found or computation failed")
        
        return self._cache[signal_name]
    
    def get_metadata(self, signal_name: str) -> Optional[SignalMetadata]:
        """Get metadata for a computed signal."""
        if signal_name not in self._cache:
            # Trigger computation
            _ = self[signal_name]
        
        return self._metadata.get(signal_name)
    
    def compute_all(self) -> SignalResult:
        """Compute all signals and return standard result."""
        return self.computer.compute(self.data, signal_names=self.signal_names)
    
    @property
    def available_signals(self) -> List[str]:
        """List of available signals."""
        if self.signal_names:
            return self.signal_names
        else:
            return list(self.computer._signals.keys())