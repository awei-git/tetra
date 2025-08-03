"""Configuration for signal computation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import pandas as pd


@dataclass
class SignalConfig:
    """Configuration for signal computation."""
    
    # Technical indicator settings
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50, 200])
    sma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    atr_period: int = 14
    adx_period: int = 14
    stoch_period: int = 14
    stoch_smooth_k: int = 3
    stoch_smooth_d: int = 3
    cci_period: int = 20
    roc_period: int = 10
    williams_r_period: int = 14
    mfi_period: int = 14
    obv_ema_period: int = 20
    vwap_period: int = 14
    pivot_support_resistance_levels: int = 3
    
    # Statistical settings
    returns_periods: List[int] = field(default_factory=lambda: [1, 5, 20, 60])
    volatility_window: int = 20
    correlation_window: int = 60
    beta_window: int = 60
    var_confidence: float = 0.95
    cvar_confidence: float = 0.95
    sharpe_window: int = 252
    rolling_quantiles: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    regime_lookback: int = 252
    hurst_min_lag: int = 2
    hurst_max_lag: int = 100
    
    # ML settings
    ml_feature_window: int = 20
    ml_prediction_horizon: int = 5
    ml_min_train_samples: int = 1000
    ml_retrain_frequency: int = 20
    ml_confidence_threshold: float = 0.6
    
    # Performance settings
    parallel_compute: bool = True
    cache_results: bool = True
    cache_ttl_seconds: int = 300
    batch_size: int = 1000
    
    # Signal selection
    compute_technical: bool = True
    compute_statistical: bool = True
    compute_ml: bool = True
    technical_signals: Optional[Set[str]] = None  # None means all
    statistical_signals: Optional[Set[str]] = None
    ml_signals: Optional[Set[str]] = None
    
    # Data validation
    min_data_points: int = 200
    max_missing_pct: float = 0.1
    handle_missing: str = "interpolate"  # "drop", "interpolate", "forward_fill"
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.min_data_points < 50:
            raise ValueError("min_data_points must be at least 50")
        
        if self.max_missing_pct < 0 or self.max_missing_pct > 1:
            raise ValueError("max_missing_pct must be between 0 and 1")
        
        if self.handle_missing not in ["drop", "interpolate", "forward_fill"]:
            raise ValueError("handle_missing must be one of: drop, interpolate, forward_fill")
        
        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")
        
        # Validate indicator periods
        for period in [self.rsi_period, self.bb_period, self.atr_period, 
                      self.adx_period, self.stoch_period, self.cci_period]:
            if period < 2:
                raise ValueError(f"Indicator period {period} must be at least 2")
        
        # Validate MACD
        if self.macd_fast >= self.macd_slow:
            raise ValueError("MACD fast period must be less than slow period")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            # Technical
            'rsi_period': self.rsi_period,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'ema_periods': self.ema_periods,
            'sma_periods': self.sma_periods,
            'atr_period': self.atr_period,
            'adx_period': self.adx_period,
            'stoch_period': self.stoch_period,
            'stoch_smooth_k': self.stoch_smooth_k,
            'stoch_smooth_d': self.stoch_smooth_d,
            'cci_period': self.cci_period,
            'roc_period': self.roc_period,
            'williams_r_period': self.williams_r_period,
            'mfi_period': self.mfi_period,
            'obv_ema_period': self.obv_ema_period,
            'vwap_period': self.vwap_period,
            'pivot_support_resistance_levels': self.pivot_support_resistance_levels,
            
            # Statistical
            'returns_periods': self.returns_periods,
            'volatility_window': self.volatility_window,
            'correlation_window': self.correlation_window,
            'beta_window': self.beta_window,
            'var_confidence': self.var_confidence,
            'cvar_confidence': self.cvar_confidence,
            'sharpe_window': self.sharpe_window,
            'rolling_quantiles': self.rolling_quantiles,
            'regime_lookback': self.regime_lookback,
            'hurst_min_lag': self.hurst_min_lag,
            'hurst_max_lag': self.hurst_max_lag,
            
            # ML
            'ml_feature_window': self.ml_feature_window,
            'ml_prediction_horizon': self.ml_prediction_horizon,
            'ml_min_train_samples': self.ml_min_train_samples,
            'ml_retrain_frequency': self.ml_retrain_frequency,
            'ml_confidence_threshold': self.ml_confidence_threshold,
            
            # Performance
            'parallel_compute': self.parallel_compute,
            'cache_results': self.cache_results,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'batch_size': self.batch_size,
            
            # Selection
            'compute_technical': self.compute_technical,
            'compute_statistical': self.compute_statistical,
            'compute_ml': self.compute_ml,
            'technical_signals': list(self.technical_signals) if self.technical_signals else None,
            'statistical_signals': list(self.statistical_signals) if self.statistical_signals else None,
            'ml_signals': list(self.ml_signals) if self.ml_signals else None,
            
            # Validation
            'min_data_points': self.min_data_points,
            'max_missing_pct': self.max_missing_pct,
            'handle_missing': self.handle_missing,
            
            # Custom
            'custom_params': self.custom_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalConfig':
        """Create from dictionary."""
        # Convert lists back to sets for signal selection
        if 'technical_signals' in data and data['technical_signals'] is not None:
            data['technical_signals'] = set(data['technical_signals'])
        if 'statistical_signals' in data and data['statistical_signals'] is not None:
            data['statistical_signals'] = set(data['statistical_signals'])
        if 'ml_signals' in data and data['ml_signals'] is not None:
            data['ml_signals'] = set(data['ml_signals'])
        
        return cls(**data)
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get parameter value."""
        # Check standard params first
        if hasattr(self, key):
            return getattr(self, key)
        
        # Check custom params
        return self.custom_params.get(key, default)