"""Regression-based ML signals."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
import xgboost as xgb

from .base import MLSignalBase
from ..base import SignalType
from ..base.config import SignalConfig


class PriceRegressionSignal(MLSignalBase):
    """Predicts future price using ML regression."""
    
    def __init__(self, config: SignalConfig,
                 feature_window: Optional[int] = None,
                 prediction_horizon: Optional[int] = None):
        super().__init__(config, feature_window, prediction_horizon)
    
    @property
    def name(self) -> str:
        return f"PriceRegression_{self.prediction_horizon}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_REGRESSION
    
    @property
    def description(self) -> str:
        return f"ML price regression with {self.prediction_horizon}-period horizon"
    
    @property
    def dependencies(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for price prediction."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        returns = close.pct_change()
        
        # Lagged returns
        for lag in range(1, self.feature_window + 1):
            features[f'return_lag_{lag}'] = returns.shift(lag)
        
        # Technical indicators
        for period in [5, 10, 20, 50]:
            # Moving averages
            sma = close.rolling(period).mean()
            features[f'sma_{period}_ratio'] = close / sma - 1
            
            # Exponential moving average
            ema = close.ewm(span=period).mean()
            features[f'ema_{period}_ratio'] = close / ema - 1
            
            # Volatility
            features[f'volatility_{period}'] = returns.rolling(period).std()
            
            # Price channels
            features[f'high_ratio_{period}'] = high.rolling(period).max() / close - 1
            features[f'low_ratio_{period}'] = close / low.rolling(period).min() - 1
        
        # Volume features
        features['volume_sma_ratio'] = volume / volume.rolling(20).mean()
        features['dollar_volume'] = close * volume
        features['dollar_volume_ratio'] = features['dollar_volume'] / \
                                         features['dollar_volume'].rolling(20).mean()
        
        # Momentum indicators
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Market microstructure
        features['spread'] = (high - low) / close
        features['close_location'] = (close - low) / (high - low)
        
        # Time features
        if hasattr(data.index, 'dayofweek'):
            features['day_of_week'] = data.index.dayofweek
            features['day_of_month'] = data.index.day
            features['month'] = data.index.month
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create labels: future returns."""
        close = data['close']
        
        # Calculate future returns
        future_returns = (close.shift(-self.prediction_horizon) / close - 1) * 100
        
        return future_returns
    
    def create_model(self) -> Any:
        """Create XGBoost regressor."""
        return xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )


class VolatilityRegressionSignal(MLSignalBase):
    """Predicts future volatility using ML regression."""
    
    def __init__(self, config: SignalConfig,
                 vol_window: int = 20):
        super().__init__(config)
        self.vol_window = vol_window
    
    @property
    def name(self) -> str:
        return f"VolatilityRegression_{self.vol_window}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_REGRESSION
    
    @property
    def description(self) -> str:
        return f"ML volatility regression for {self.vol_window}-day volatility"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-specific features."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data['high']
        low = data['low']
        returns = close.pct_change()
        
        # Historical volatility measures
        for period in [5, 10, 20, 50]:
            # Standard deviation
            features[f'hist_vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)
            
            # Parkinson volatility
            hl_ratio = np.log(high / low)
            features[f'parkinson_vol_{period}'] = np.sqrt(
                hl_ratio.rolling(period).apply(lambda x: np.sum(x**2) / (4 * np.log(2) * len(x)))
            ) * np.sqrt(252)
            
            # Garman-Klass volatility
            co_ratio = np.log(close / close.shift(1))
            features[f'gk_vol_{period}'] = np.sqrt(
                0.5 * hl_ratio**2 - (2 * np.log(2) - 1) * co_ratio**2
            ).rolling(period).mean() * np.sqrt(252)
        
        # Volatility of volatility
        vol_series = returns.rolling(20).std()
        features['vol_of_vol'] = vol_series.rolling(20).std()
        
        # GARCH-like features
        features['squared_returns'] = returns**2
        for lag in range(1, 6):
            features[f'squared_returns_lag_{lag}'] = features['squared_returns'].shift(lag)
            features[f'vol_lag_{lag}'] = vol_series.shift(lag)
        
        # Volatility regime features
        features['vol_percentile'] = vol_series.rolling(252).rank(pct=True)
        features['vol_zscore'] = (vol_series - vol_series.rolling(252).mean()) / \
                                vol_series.rolling(252).std()
        
        # Volume-volatility interaction
        features['volume_volatility'] = data['volume'].rolling(20).std() / \
                                       data['volume'].rolling(20).mean()
        
        # Intraday range
        features['intraday_range'] = (high - low) / close
        features['avg_intraday_range'] = features['intraday_range'].rolling(20).mean()
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create labels: future realized volatility."""
        returns = data['close'].pct_change()
        
        # Calculate future realized volatility
        future_vol = returns.shift(-self.vol_window).rolling(self.vol_window).std() * np.sqrt(252) * 100
        
        return future_vol
    
    def create_model(self) -> Any:
        """Create Gradient Boosting regressor."""
        return GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )


class ReturnsPredictionSignal(MLSignalBase):
    """Predicts returns distribution parameters."""
    
    def __init__(self, config: SignalConfig,
                 return_type: str = 'mean'):
        super().__init__(config)
        self.return_type = return_type  # 'mean', 'median', 'sharpe'
    
    @property
    def name(self) -> str:
        return f"ReturnsPrediction_{self.return_type}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_REGRESSION
    
    @property
    def description(self) -> str:
        return f"ML {self.return_type} returns prediction"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create return prediction features."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()
        
        # Return statistics
        for period in [5, 10, 20, 50]:
            features[f'mean_return_{period}'] = returns.rolling(period).mean()
            features[f'std_return_{period}'] = returns.rolling(period).std()
            features[f'skew_return_{period}'] = returns.rolling(period).skew()
            features[f'kurt_return_{period}'] = returns.rolling(period).kurt()
            
            # Sharpe ratio
            features[f'sharpe_{period}'] = features[f'mean_return_{period}'] / \
                                          features[f'std_return_{period}'] * np.sqrt(252)
        
        # Autocorrelation features
        for lag in [1, 5, 10, 20]:
            features[f'return_autocorr_{lag}'] = returns.rolling(50).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        # Volume-return interaction
        features['volume_return_corr'] = returns.rolling(20).corr(volume.pct_change())
        
        # Trend features
        for period in [20, 50]:
            features[f'trend_{period}'] = close.rolling(period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean()
            )
        
        # Mean reversion features
        for period in [20, 50]:
            sma = close.rolling(period).mean()
            features[f'mean_reversion_{period}'] = (close - sma) / sma
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create labels based on return type."""
        returns = data['close'].pct_change()
        
        if self.return_type == 'mean':
            # Future mean return
            labels = returns.shift(-self.prediction_horizon).rolling(
                self.prediction_horizon
            ).mean() * 100
        elif self.return_type == 'median':
            # Future median return
            labels = returns.shift(-self.prediction_horizon).rolling(
                self.prediction_horizon
            ).median() * 100
        elif self.return_type == 'sharpe':
            # Future Sharpe ratio
            future_mean = returns.shift(-self.prediction_horizon).rolling(
                self.prediction_horizon
            ).mean()
            future_std = returns.shift(-self.prediction_horizon).rolling(
                self.prediction_horizon
            ).std()
            labels = (future_mean / future_std) * np.sqrt(252)
        else:
            raise ValueError(f"Unknown return type: {self.return_type}")
        
        return labels
    
    def create_model(self) -> Any:
        """Create Ridge regression model."""
        return Ridge(
            alpha=1.0,
            fit_intercept=True,
            random_state=42
        )


class MultiFactorRegressionSignal(MLSignalBase):
    """Multi-factor model for returns prediction."""
    
    def __init__(self, config: SignalConfig):
        super().__init__(config)
    
    @property
    def name(self) -> str:
        return "MultiFactorRegression"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_REGRESSION
    
    @property
    def description(self) -> str:
        return "Multi-factor ML regression model"
    
    @property
    def dependencies(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create multi-factor features."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        returns = close.pct_change()
        
        # Value factor
        features['value_factor'] = close.rolling(252).mean() / close
        
        # Momentum factors
        for period in [20, 50, 100, 200]:
            features[f'momentum_{period}'] = close / close.shift(period) - 1
        
        # Size factor (using volume as proxy)
        features['size_factor'] = np.log(volume.rolling(20).mean())
        
        # Volatility factor
        features['volatility_factor'] = returns.rolling(20).std() * np.sqrt(252)
        
        # Quality factors
        # Efficiency ratio
        net_change = close - close.shift(20)
        sum_changes = close.diff().abs().rolling(20).sum()
        features['efficiency_ratio'] = net_change / sum_changes
        
        # Consistency of returns
        features['return_consistency'] = returns.rolling(50).apply(
            lambda x: np.sum(x > 0) / len(x)
        )
        
        # Low volatility factor
        features['low_vol_factor'] = 1 / (returns.rolling(50).std() + 0.001)
        
        # Mean reversion factor
        sma_50 = close.rolling(50).mean()
        features['mean_reversion_factor'] = (sma_50 - close) / sma_50
        
        # Liquidity factor
        features['liquidity_factor'] = volume / volume.rolling(50).mean()
        
        # Market regime factors
        market_vol = returns.rolling(100).std() * np.sqrt(252)
        features['vol_regime'] = market_vol / market_vol.rolling(252).mean()
        
        # Trend strength
        features['trend_strength'] = close.rolling(50).apply(
            lambda x: abs(np.polyfit(range(len(x)), x, 1)[0]) / x.std()
        )
        
        # Interaction terms
        features['momentum_volatility'] = features['momentum_50'] * features['volatility_factor']
        features['value_momentum'] = features['value_factor'] * features['momentum_50']
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create labels: risk-adjusted returns."""
        returns = data['close'].pct_change()
        
        # Calculate future risk-adjusted returns
        future_returns = returns.shift(-self.prediction_horizon).rolling(
            self.prediction_horizon
        ).mean()
        future_vol = returns.shift(-self.prediction_horizon).rolling(
            self.prediction_horizon
        ).std()
        
        # Risk-adjusted return (Sharpe-like)
        risk_adj_returns = (future_returns / (future_vol + 0.001)) * 100
        
        return risk_adj_returns
    
    def create_model(self) -> Any:
        """Create ElasticNet model for factor selection."""
        return ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=1000,
            random_state=42
        )