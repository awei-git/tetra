"""Batch signal computer for efficient multi-symbol computation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
import logging

from ..base import SignalComputer, SignalConfig, SignalResult
from ..base.base_signal import BaseSignal

logger = logging.getLogger(__name__)


class BatchSignalComputer(SignalComputer):
    """Optimized signal computer for batch processing multiple symbols."""
    
    def __init__(self, 
                 config: Optional[SignalConfig] = None,
                 n_processes: Optional[int] = None):
        super().__init__(config)
        
        # Determine number of processes
        if n_processes is None:
            n_processes = min(mp.cpu_count() - 1, 8)
        self.n_processes = max(1, n_processes)
        
        # Process pool for multi-symbol computation
        self._process_pool = None
        if self.n_processes > 1:
            self._process_pool = ProcessPoolExecutor(max_workers=self.n_processes)
    
    def compute_batch(self,
                     data: pd.DataFrame,
                     symbols: List[str],
                     signal_names: Optional[List[str]] = None,
                     chunk_size: Optional[int] = None) -> Dict[str, SignalResult]:
        """Compute signals for multiple symbols in batch.
        
        Args:
            data: DataFrame with OHLCV data and 'symbol' column
            symbols: List of symbols to process
            signal_names: Signals to compute
            chunk_size: Number of symbols per chunk
            
        Returns:
            Dictionary mapping symbol to SignalResult
        """
        start_time = time.time()
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = max(1, len(symbols) // (self.n_processes * 2))
        
        # Create symbol chunks
        symbol_chunks = [symbols[i:i+chunk_size] 
                        for i in range(0, len(symbols), chunk_size)]
        
        logger.info(f"Processing {len(symbols)} symbols in {len(symbol_chunks)} chunks")
        
        # Process chunks
        results = {}
        
        if self._process_pool and len(symbol_chunks) > 1:
            # Parallel processing
            futures = {}
            for chunk in symbol_chunks:
                # Extract data for this chunk
                chunk_data = data[data['symbol'].isin(chunk)]
                
                future = self._process_pool.submit(
                    self._process_chunk,
                    chunk_data,
                    chunk,
                    signal_names
                )
                futures[future] = chunk
            
            # Collect results
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    chunk_results = future.result()
                    results.update(chunk_results)
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk}: {e}")
                    # Add error results for failed symbols
                    for symbol in chunk:
                        results[symbol] = SignalResult(
                            data=pd.DataFrame(),
                            metadata={},
                            compute_time=0,
                            errors={'batch_error': str(e)},
                            warnings=[]
                        )
        else:
            # Sequential processing
            for chunk in symbol_chunks:
                chunk_data = data[data['symbol'].isin(chunk)]
                chunk_results = self._process_chunk(chunk_data, chunk, signal_names)
                results.update(chunk_results)
        
        total_time = time.time() - start_time
        logger.info(f"Batch computation completed in {total_time:.2f}s")
        
        return results
    
    def _process_chunk(self,
                      data: pd.DataFrame,
                      symbols: List[str],
                      signal_names: Optional[List[str]]) -> Dict[str, SignalResult]:
        """Process a chunk of symbols."""
        results = {}
        
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol].drop(columns=['symbol'])
            
            if len(symbol_data) == 0:
                results[symbol] = SignalResult(
                    data=pd.DataFrame(),
                    metadata={},
                    compute_time=0,
                    errors={'no_data': f"No data for symbol {symbol}"},
                    warnings=[]
                )
                continue
            
            # Compute signals for this symbol
            result = self.compute(
                symbol_data,
                signal_names=signal_names
            )
            results[symbol] = result
        
        return results
    
    def compute_optimized(self,
                         data: pd.DataFrame,
                         signal_names: Optional[List[str]] = None) -> SignalResult:
        """Compute signals with optimizations for multi-symbol data.
        
        This method uses vectorized operations where possible to compute
        signals across all symbols simultaneously.
        """
        start_time = time.time()
        
        # Check if we have multi-symbol data
        if 'symbol' not in data.columns:
            # Single symbol, use standard computation
            return self.compute(data, signal_names=signal_names)
        
        # Get unique symbols
        symbols = data['symbol'].unique()
        
        # Determine which signals to compute
        signals_to_compute = self._get_signals_to_compute(signal_names)
        
        # Group signals by type for vectorized computation
        vectorizable_signals = []
        non_vectorizable_signals = []
        
        for signal_name in signals_to_compute:
            signal = self._signals.get(signal_name)
            if signal and hasattr(signal, 'compute_vectorized'):
                vectorizable_signals.append(signal_name)
            else:
                non_vectorizable_signals.append(signal_name)
        
        logger.info(f"Vectorizable signals: {len(vectorizable_signals)}, "
                   f"Non-vectorizable: {len(non_vectorizable_signals)}")
        
        # Prepare result containers
        all_results = {}
        all_metadata = {}
        all_errors = {}
        all_warnings = []
        
        # Compute vectorizable signals
        if vectorizable_signals:
            vec_result = self._compute_vectorized_signals(
                data, symbols, vectorizable_signals
            )
            for symbol in symbols:
                all_results[symbol] = vec_result.get(symbol, pd.DataFrame())
                all_metadata[symbol] = {}
        
        # Compute non-vectorizable signals using batch processing
        if non_vectorizable_signals:
            batch_results = self.compute_batch(
                data, 
                list(symbols),
                signal_names=non_vectorizable_signals
            )
            
            # Merge results
            for symbol, result in batch_results.items():
                if symbol in all_results:
                    all_results[symbol] = pd.concat([
                        all_results[symbol], 
                        result.data
                    ], axis=1)
                else:
                    all_results[symbol] = result.data
                
                all_metadata[symbol].update(result.metadata)
                
                for error_key, error_msg in result.errors.items():
                    all_errors[f"{symbol}_{error_key}"] = error_msg
                
                for warning in result.warnings:
                    all_warnings.append(f"{symbol}: {warning}")
        
        # Combine all results into single DataFrame
        combined_data = pd.DataFrame()
        for symbol, symbol_data in all_results.items():
            # Add symbol prefix to columns
            symbol_data.columns = [f"{symbol}_{col}" for col in symbol_data.columns]
            if combined_data.empty:
                combined_data = symbol_data
            else:
                combined_data = pd.concat([combined_data, symbol_data], axis=1)
        
        # Flatten metadata
        flat_metadata = {}
        for symbol, meta_dict in all_metadata.items():
            for signal_name, meta in meta_dict.items():
                flat_metadata[f"{symbol}_{signal_name}"] = meta
        
        compute_time = time.time() - start_time
        
        return SignalResult(
            data=combined_data,
            metadata=flat_metadata,
            compute_time=compute_time,
            errors=all_errors,
            warnings=all_warnings
        )
    
    def _compute_vectorized_signals(self,
                                   data: pd.DataFrame,
                                   symbols: List[str],
                                   signal_names: List[str]) -> Dict[str, pd.DataFrame]:
        """Compute signals that support vectorized operations."""
        # Pivot data for vectorized computation
        pivoted_data = {}
        
        # Pivot each OHLCV column
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                pivoted_data[col] = data.pivot(
                    index='date' if 'date' in data.columns else data.index,
                    columns='symbol',
                    values=col
                )[symbols]
        
        # Compute each vectorized signal
        results = {symbol: pd.DataFrame(index=pivoted_data['close'].index) 
                  for symbol in symbols}
        
        for signal_name in signal_names:
            signal = self._signals[signal_name]
            try:
                # Call vectorized compute method
                signal_results = signal.compute_vectorized(pivoted_data)
                
                # Assign results to each symbol
                for symbol in symbols:
                    if symbol in signal_results.columns:
                        results[symbol][signal_name] = signal_results[symbol]
                    
            except Exception as e:
                logger.error(f"Error in vectorized computation of {signal_name}: {e}")
        
        return results
    
    def close(self):
        """Close process pool."""
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
        super().__del__()