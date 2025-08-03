"""Classification-based ML signals."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from .base import MLSignalBase
from ..base import SignalType
from ..base.config import SignalConfig


class DirectionClassifierSignal(MLSignalBase):
    """Predicts price direction (up/down/neutral) using ML classification."""
    
    def __init__(self, config: SignalConfig,
                 feature_window: Optional[int] = None,
                 prediction_horizon: Optional[int] = None,
                 threshold: float = 0.001):
        super().__init__(config, feature_window, prediction_horizon)
        self.threshold = threshold  # Threshold for neutral class
    
    @property
    def name(self) -> str:
        return f"DirectionClassifier_{self.prediction_horizon}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_CLASSIFICATION
    
    @property
    def description(self) -> str:
        return f"ML price direction classifier with {self.prediction_horizon}-period horizon"
    
    @property
    def dependencies(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical features for classification."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Price-based features
        for period in [5, 10, 20]:
            # Returns
            features[f'return_{period}'] = close.pct_change(period)
            
            # Moving averages
            features[f'sma_{period}'] = close.rolling(period).mean() / close - 1
            
            # Volatility
            features[f'volatility_{period}'] = close.pct_change().rolling(period).std()
            
            # Price position
            features[f'price_position_{period}'] = (close - close.rolling(period).min()) / \
                                                  (close.rolling(period).max() - close.rolling(period).min())
        
        # Volume features
        features['volume_ratio'] = volume / volume.rolling(20).mean()
        features['volume_trend'] = volume.rolling(5).mean() / volume.rolling(20).mean()
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        features['rsi'] = 100 - (100 / (1 + gain / loss))
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        features['macd'] = (ema12 - ema26) / close
        
        # Bollinger Bands position
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features['bb_position'] = (close - sma20) / (2 * std20)
        
        # High-Low spread
        features['hl_spread'] = (high - low) / close
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create labels: -1 (down), 0 (neutral), 1 (up)."""
        close = data['close']
        
        # Calculate future returns
        future_returns = close.shift(-self.prediction_horizon) / close - 1
        
        # Classify into three categories
        labels = pd.Series(0, index=data.index)  # Default neutral
        labels[future_returns > self.threshold] = 1   # Up
        labels[future_returns < -self.threshold] = -1  # Down
        
        return labels
    
    def create_model(self) -> Any:
        """Create RandomForest classifier."""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Override to return probabilities instead of classes."""
        if not self.is_trained or self.model is None:
            raise ValueError(f"{self.name} model not trained")
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get probability predictions
        probas = self.model.predict_proba(features_scaled)
        
        # Calculate directional score: P(up) - P(down)
        # Assuming classes are ordered: -1, 0, 1
        if self.model.classes_[0] == -1:  # [-1, 0, 1]
            score = probas[:, 2] - probas[:, 0]  # P(up) - P(down)
        else:  # Handle different class ordering
            up_idx = list(self.model.classes_).index(1)
            down_idx = list(self.model.classes_).index(-1)
            score = probas[:, up_idx] - probas[:, down_idx]
        
        # Scale to -100 to 100
        score = score * 100
        
        return pd.Series(score, index=features.index)


class RegimeClassifierSignal(MLSignalBase):
    """Classifies market regime using ML."""
    
    def __init__(self, config: SignalConfig,
                 feature_window: Optional[int] = None,
                 n_regimes: int = 4):
        super().__init__(config, feature_window)
        self.n_regimes = n_regimes
    
    @property
    def name(self) -> str:
        return f"RegimeClassifier_{self.n_regimes}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_CLASSIFICATION
    
    @property
    def description(self) -> str:
        return f"ML market regime classifier with {self.n_regimes} regimes"
    
    @property
    def dependencies(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create regime-related features."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()
        
        # Trend features
        for period in [20, 50, 100]:
            features[f'trend_{period}'] = close / close.rolling(period).mean() - 1
            features[f'trend_strength_{period}'] = (close - close.rolling(period).min()) / \
                                                  (close.rolling(period).max() - close.rolling(period).min())
        
        # Volatility regime features
        for period in [10, 20, 50]:
            features[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
            features[f'vol_ratio_{period}'] = returns.rolling(period).std() / returns.rolling(period * 2).std()
        
        # Volume regime features
        features['volume_ma_ratio'] = volume / volume.rolling(50).mean()
        features['volume_trend'] = volume.rolling(20).mean() / volume.rolling(50).mean()
        
        # Market microstructure
        features['efficiency_ratio'] = abs(close - close.shift(20)) / \
                                      (close.diff().abs().rolling(20).sum())
        
        # Correlation features
        for lag in [5, 10, 20]:
            features[f'autocorr_{lag}'] = returns.rolling(50).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        # Distribution features
        features['skewness'] = returns.rolling(50).skew()
        features['kurtosis'] = returns.rolling(50).kurt()
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create regime labels using volatility and trend."""
        close = data['close']
        returns = close.pct_change()
        
        # Calculate rolling metrics
        vol = returns.rolling(20).std() * np.sqrt(252)
        trend = close.rolling(50).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Normalize
        vol_norm = (vol - vol.rolling(252).mean()) / vol.rolling(252).std()
        trend_norm = (trend - trend.rolling(252).mean()) / trend.rolling(252).std()
        
        # Create regime labels
        labels = pd.Series(0, index=data.index)
        
        # Define regimes based on volatility and trend
        labels[(vol_norm > 0.5) & (trend_norm > 0.5)] = 0   # High vol, uptrend
        labels[(vol_norm > 0.5) & (trend_norm <= 0.5)] = 1  # High vol, downtrend
        labels[(vol_norm <= 0.5) & (trend_norm > 0.5)] = 2  # Low vol, uptrend
        labels[(vol_norm <= 0.5) & (trend_norm <= 0.5)] = 3 # Low vol, downtrend
        
        return labels
    
    def create_model(self) -> Any:
        """Create Gradient Boosting classifier."""
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )


class PatternClassifierSignal(MLSignalBase):
    """Classifies chart patterns using ML."""
    
    def __init__(self, config: SignalConfig,
                 pattern_window: int = 20):
        super().__init__(config, feature_window=pattern_window)
        self.pattern_window = pattern_window
    
    @property
    def name(self) -> str:
        return f"PatternClassifier_{self.pattern_window}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_CLASSIFICATION
    
    @property
    def description(self) -> str:
        return f"ML chart pattern classifier with {self.pattern_window} period window"
    
    @property
    def dependencies(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create pattern-based features."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Normalized price series for pattern recognition
        for i in range(1, self.pattern_window + 1):
            features[f'close_lag_{i}'] = close.shift(i) / close - 1
            features[f'high_lag_{i}'] = high.shift(i) / close - 1
            features[f'low_lag_{i}'] = low.shift(i) / close - 1
            features[f'volume_lag_{i}'] = volume.shift(i) / volume.rolling(20).mean()
        
        # Shape features
        window_close = pd.DataFrame({
            f'c_{i}': close.shift(i) for i in range(self.pattern_window)
        })
        
        # Linear regression slope
        features['trend_slope'] = window_close.apply(
            lambda row: np.polyfit(range(len(row)), row.values, 1)[0] 
            if not row.isna().any() else np.nan,
            axis=1
        )
        
        # Curvature (2nd order polynomial)
        features['curvature'] = window_close.apply(
            lambda row: np.polyfit(range(len(row)), row.values, 2)[0] 
            if not row.isna().any() else np.nan,
            axis=1
        )
        
        # Peak/trough detection
        features['n_peaks'] = window_close.apply(
            lambda row: self._count_peaks(row.values) if not row.isna().any() else np.nan,
            axis=1
        )
        
        return features.dropna()
    
    def _count_peaks(self, values: np.ndarray) -> int:
        """Count number of peaks in array."""
        peaks = 0
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                peaks += 1
        return peaks
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create pattern labels based on future price movement."""
        close = data['close']
        
        # Simple labeling based on significant moves
        future_return = close.shift(-5) / close - 1
        
        labels = pd.Series(0, index=data.index)  # Default: no pattern
        labels[future_return > 0.02] = 1   # Bullish pattern
        labels[future_return < -0.02] = 2  # Bearish pattern
        
        return labels
    
    def create_model(self) -> Any:
        """Create SVM classifier for pattern recognition."""
        return SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )


class SupportResistanceMLSignal(MLSignalBase):
    """ML-based support and resistance level detection."""
    
    def __init__(self, config: SignalConfig,
                 lookback: int = 50):
        super().__init__(config, feature_window=lookback)
        self.lookback = lookback
    
    @property
    def name(self) -> str:
        return "SupportResistanceML"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_CLASSIFICATION
    
    @property
    def description(self) -> str:
        return "ML-based support and resistance detection"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for S/R detection."""
        features = pd.DataFrame(index=data.index)
        
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Price levels relative to recent range
        for period in [10, 20, 50]:
            features[f'price_level_{period}'] = (close - low.rolling(period).min()) / \
                                               (high.rolling(period).max() - low.rolling(period).min())
            
            # Distance to recent highs/lows
            features[f'dist_to_high_{period}'] = (high.rolling(period).max() - close) / close
            features[f'dist_to_low_{period}'] = (close - low.rolling(period).min()) / close
        
        # Volume at price levels
        features['volume_at_high'] = volume.rolling(5).mean() / volume.rolling(20).mean()
        
        # Price reaction features
        features['bounce_strength'] = close.diff().abs() / close.shift(1)
        features['reversal_count'] = (close.diff() * close.diff().shift(1) < 0).rolling(20).sum()
        
        # Clustering of touches
        for level_pct in [0.98, 0.99, 1.01, 1.02]:
            level_touches = ((high / close > level_pct - 0.005) & 
                           (high / close < level_pct + 0.005)).rolling(20).sum()
            features[f'touches_near_{level_pct}'] = level_touches
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create labels for support/resistance."""
        close = data['close']
        high = data['high']
        low = data['low']
        
        # Label based on future price reaction
        future_high = high.shift(-5).rolling(5).max()
        future_low = low.shift(-5).rolling(5).min()
        
        labels = pd.Series(0, index=data.index)  # Default: neither
        
        # Resistance: price fails to break above
        resistance_condition = (future_high < close * 1.01) & (close > close.rolling(20).mean())
        labels[resistance_condition] = 1
        
        # Support: price fails to break below  
        support_condition = (future_low > close * 0.99) & (close < close.rolling(20).mean())
        labels[support_condition] = -1
        
        return labels
    
    def create_model(self) -> Any:
        """Create ensemble classifier."""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )