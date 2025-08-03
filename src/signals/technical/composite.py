"""Composite class that registers all technical indicators."""

from typing import Optional, List

from ..base import SignalComputer, SignalConfig
from .trend import (
    SMASignal, EMASignal, WMASignal, MACDSignal, 
    ADXSignal, ParabolicSARSignal, IchimokuSignal, SupertrendSignal
)
from .momentum import (
    RSISignal, StochasticSignal, CCISignal, ROCSignal,
    WilliamsRSignal, MomentumSignal, TSISignal, UltimateOscillatorSignal
)
from .volatility import (
    BollingerBandsSignal, ATRSignal, KeltnerChannelSignal,
    DonchianChannelSignal, StandardDeviationSignal, HistoricalVolatilitySignal
)
from .volume import (
    OBVSignal, MFISignal, VWAPSignal, ADLSignal,
    CMFSignal, VolumeProfileSignal, EaseOfMovementSignal
)
from .pattern import (
    PivotPointsSignal, FibonacciRetracementSignal,
    SupportResistanceSignal, CandlePatternSignal
)


class TechnicalSignals(SignalComputer):
    """Composite class that provides all technical indicators."""
    
    def __init__(self, config: Optional[SignalConfig] = None):
        super().__init__(config)
        self._register_all_signals()
    
    def _register_all_signals(self):
        """Register all technical indicators."""
        
        # Trend indicators
        for period in self.config.sma_periods:
            self.register_signal(SMASignal(self.config, period=period))
        
        for period in self.config.ema_periods:
            self.register_signal(EMASignal(self.config, period=period))
        
        self.register_signal(WMASignal(self.config))
        self.register_signal(MACDSignal(self.config))
        self.register_signal(ADXSignal(self.config))
        self.register_signal(ParabolicSARSignal(self.config))
        self.register_signal(IchimokuSignal(self.config))
        self.register_signal(SupertrendSignal(self.config))
        
        # Momentum indicators
        self.register_signal(RSISignal(self.config))
        self.register_signal(StochasticSignal(self.config))
        self.register_signal(CCISignal(self.config))
        self.register_signal(ROCSignal(self.config))
        self.register_signal(WilliamsRSignal(self.config))
        self.register_signal(MomentumSignal(self.config))
        self.register_signal(TSISignal(self.config))
        self.register_signal(UltimateOscillatorSignal(self.config))
        
        # Volatility indicators
        self.register_signal(BollingerBandsSignal(self.config))
        self.register_signal(ATRSignal(self.config))
        self.register_signal(KeltnerChannelSignal(self.config))
        self.register_signal(DonchianChannelSignal(self.config))
        self.register_signal(StandardDeviationSignal(self.config))
        self.register_signal(HistoricalVolatilitySignal(self.config))
        
        # Volume indicators
        self.register_signal(OBVSignal(self.config))
        self.register_signal(MFISignal(self.config))
        self.register_signal(VWAPSignal(self.config))
        self.register_signal(ADLSignal(self.config))
        self.register_signal(CMFSignal(self.config))
        self.register_signal(VolumeProfileSignal(self.config))
        self.register_signal(EaseOfMovementSignal(self.config))
        
        # Pattern indicators
        self.register_signal(PivotPointsSignal(self.config))
        self.register_signal(FibonacciRetracementSignal(self.config))
        self.register_signal(SupportResistanceSignal(self.config))
        self.register_signal(CandlePatternSignal(self.config))
    
    def get_trend_signals(self) -> List[str]:
        """Get list of trend indicator names."""
        return [
            f"SMA_{period}" for period in self.config.sma_periods
        ] + [
            f"EMA_{period}" for period in self.config.ema_periods
        ] + [
            "WMA_20", "MACD", "ADX", "PSAR", "Ichimoku", "Supertrend"
        ]
    
    def get_momentum_signals(self) -> List[str]:
        """Get list of momentum indicator names."""
        return [
            f"RSI_{self.config.rsi_period}",
            "Stochastic", "CCI", f"ROC_{self.config.roc_period}",
            "WilliamsR", "Momentum_10", "TSI", "UltimateOsc"
        ]
    
    def get_volatility_signals(self) -> List[str]:
        """Get list of volatility indicator names."""
        return [
            "BollingerBands", f"ATR_{self.config.atr_period}",
            "KeltnerChannel", "DonchianChannel_20",
            "StdDev_20", "HV_20"
        ]
    
    def get_volume_signals(self) -> List[str]:
        """Get list of volume indicator names."""
        return [
            "OBV", "MFI", "VWAP", "ADL", "CMF",
            "VolumeProfile", "EOM"
        ]
    
    def get_pattern_signals(self) -> List[str]:
        """Get list of pattern indicator names."""
        return [
            "PivotPoints", "FibonacciRetracement",
            "SupportResistance", "CandlePattern"
        ]
    
    def get_all_technical_signals(self) -> List[str]:
        """Get list of all technical indicator names."""
        return (
            self.get_trend_signals() +
            self.get_momentum_signals() +
            self.get_volatility_signals() +
            self.get_volume_signals() +
            self.get_pattern_signals()
        )