"""Anomaly detection ML signals."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture

from .base import MLSignalBase
from ..base import SignalType
from ..base.config import SignalConfig


class PriceAnomalySignal(MLSignalBase):
    """Detects price anomalies using ML."""
    
    def __init__(self, config: SignalConfig,
                 contamination: float = 0.1,
                 method: str = 'isolation_forest'):
        super().__init__(config)
        self.contamination = contamination
        self.method = method
    
    @property
    def name(self) -> str:
        return f"PriceAnomaly_{self.method}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_ANOMALY
    
    @property
    def description(self) -> str:
        return f"Price anomaly detection using {self.method}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for anomaly detection."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        returns = close.pct_change()
        
        # Price-based features
        features['returns'] = returns
        features['abs_returns'] = returns.abs()
        features['log_returns'] = np.log(close / close.shift(1))
        
        # Standardized price moves
        for period in [5, 10, 20]:
            rolling_mean = returns.rolling(period).mean()
            rolling_std = returns.rolling(period).std()
            features[f'z_score_{period}'] = (returns - rolling_mean) / rolling_std
        
        # Price gaps
        features['gap_open'] = (data['open'] - close.shift(1)) / close.shift(1)
        features['intraday_range'] = (high - low) / close
        
        # Volume anomalies
        features['volume_ratio'] = volume / volume.rolling(20).mean()
        features['dollar_volume'] = close * volume
        features['dollar_volume_ratio'] = features['dollar_volume'] / \
                                         features['dollar_volume'].rolling(20).mean()
        
        # Price efficiency
        features['efficiency_ratio'] = close.diff().abs() / \
                                      close.diff().abs().rolling(10).sum()
        
        # Relative position in range
        features['close_location'] = (close - low) / (high - low + 0.0001)
        
        # Unusual patterns
        features['consecutive_moves'] = (returns > 0).astype(int).groupby(
            (returns > 0).astype(int).diff().ne(0).cumsum()
        ).cumsum()
        
        # Distance from moving averages
        for period in [20, 50]:
            ma = close.rolling(period).mean()
            features[f'dist_from_ma_{period}'] = (close - ma) / ma
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """No labels needed for unsupervised anomaly detection."""
        return pd.Series(index=data.index)  # Return empty series
    
    def create_model(self) -> Any:
        """Create anomaly detection model."""
        if self.method == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
        elif self.method == 'one_class_svm':
            return OneClassSVM(
                gamma='scale',
                nu=self.contamination
            )
        elif self.method == 'elliptic_envelope':
            return EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def train(self, features: pd.DataFrame, labels: pd.Series):
        """Override train for unsupervised learning."""
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        if len(features) < self.config.ml_min_train_samples:
            raise ValueError(f"Insufficient training samples: {len(features)}")
        
        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Create and train model
        self.model = self.create_model()
        self.model.fit(features_scaled)
        
        # Calculate training metrics
        anomaly_scores = self.model.decision_function(features_scaled)
        self.train_metrics = {
            'train_samples': len(features),
            'anomaly_rate': np.mean(self.model.predict(features_scaled) == -1),
            'score_mean': np.mean(anomaly_scores),
            'score_std': np.std(anomaly_scores)
        }
        
        self.is_trained = True
        self.last_train_date = features.index[-1]
        
        logger.info(f"{self.name} trained with {len(features)} samples")
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return anomaly scores."""
        if not self.is_trained or self.model is None:
            raise ValueError(f"{self.name} model not trained")
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get anomaly scores
        scores = self.model.decision_function(features_scaled)
        
        # Normalize scores to -100 to 100 range
        # Negative scores indicate anomalies
        normalized_scores = -scores * 100  # Invert so anomalies are positive
        
        return pd.Series(normalized_scores, index=features.index)


class VolumeAnomalySignal(MLSignalBase):
    """Detects volume anomalies."""
    
    def __init__(self, config: SignalConfig,
                 lookback: int = 50):
        super().__init__(config, feature_window=lookback)
        self.lookback = lookback
    
    @property
    def name(self) -> str:
        return "VolumeAnomaly"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_ANOMALY
    
    @property
    def description(self) -> str:
        return "Volume anomaly detection"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volume-specific features."""
        features = pd.DataFrame(index=data.index)
        
        volume = data['volume']
        close = data['close']
        returns = close.pct_change()
        
        # Volume ratios
        for period in [5, 10, 20, 50]:
            features[f'volume_ma_ratio_{period}'] = volume / volume.rolling(period).mean()
            features[f'volume_std_ratio_{period}'] = volume / volume.rolling(period).std()
        
        # Volume momentum
        features['volume_momentum'] = volume / volume.shift(1) - 1
        features['volume_acceleration'] = features['volume_momentum'].diff()
        
        # Dollar volume
        dollar_volume = close * volume
        features['dollar_volume_ratio'] = dollar_volume / dollar_volume.rolling(20).mean()
        
        # Volume-price divergence
        features['volume_return_ratio'] = (volume.pct_change() / 
                                          (returns.abs() + 0.0001))
        
        # Volume concentration
        features['volume_concentration'] = volume.rolling(5).sum() / \
                                          volume.rolling(20).sum()
        
        # Unusual volume patterns
        features['volume_spike'] = (volume > volume.rolling(20).mean() + 
                                   2 * volume.rolling(20).std()).astype(int)
        
        features['consecutive_high_volume'] = features['volume_spike'].groupby(
            (features['volume_spike'] == 0).cumsum()
        ).cumsum()
        
        # Volume profile
        features['volume_percentile'] = volume.rolling(self.lookback).rank(pct=True)
        
        # Time-based volume patterns
        if hasattr(data.index, 'dayofweek'):
            features['is_monday'] = (data.index.dayofweek == 0).astype(int)
            features['is_friday'] = (data.index.dayofweek == 4).astype(int)
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """No labels for anomaly detection."""
        return pd.Series(index=data.index)
    
    def create_model(self) -> Any:
        """Create Local Outlier Factor model."""
        return LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True
        )


class VolatilityAnomalySignal(MLSignalBase):
    """Detects volatility regime anomalies."""
    
    def __init__(self, config: SignalConfig):
        super().__init__(config)
    
    @property
    def name(self) -> str:
        return "VolatilityAnomaly"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_ANOMALY
    
    @property
    def description(self) -> str:
        return "Volatility regime anomaly detection"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volatility features."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data['high']
        low = data['low']
        returns = close.pct_change()
        
        # Various volatility measures
        for period in [5, 10, 20]:
            # Historical volatility
            features[f'hist_vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)
            
            # Parkinson volatility
            features[f'parkinson_{period}'] = np.sqrt(
                np.log(high/low)**2 / (4*np.log(2))
            ).rolling(period).mean() * np.sqrt(252)
            
            # Volatility of volatility
            vol = returns.rolling(period).std()
            features[f'vol_of_vol_{period}'] = vol.rolling(period).std()
        
        # Volatility ratios
        features['vol_ratio_5_20'] = features['hist_vol_5'] / features['hist_vol_20']
        features['vol_ratio_10_20'] = features['hist_vol_10'] / features['hist_vol_20']
        
        # Volatility regime changes
        vol_20 = returns.rolling(20).std()
        features['vol_regime_change'] = vol_20 / vol_20.shift(20) - 1
        
        # Realized vs expected volatility
        features['vol_surprise'] = vol_20 - vol_20.rolling(60).mean()
        
        # Intraday volatility patterns
        features['intraday_vol'] = (high - low) / close
        features['overnight_gap'] = (data['open'] - close.shift(1)).abs() / close.shift(1)
        
        # Volatility clustering
        features['high_vol_days'] = (vol_20 > vol_20.rolling(60).mean() + 
                                    vol_20.rolling(60).std()).astype(int)
        features['vol_cluster'] = features['high_vol_days'].rolling(10).sum()
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """No labels for anomaly detection."""
        return pd.Series(index=data.index)
    
    def create_model(self) -> Any:
        """Create Gaussian Mixture Model for regime detection."""
        return GaussianMixture(
            n_components=3,  # Low, medium, high volatility regimes
            covariance_type='full',
            random_state=42
        )
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return anomaly scores based on probability."""
        if not self.is_trained or self.model is None:
            raise ValueError(f"{self.name} model not trained")
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get log probabilities
        log_probs = self.model.score_samples(features_scaled)
        
        # Convert to anomaly scores
        # Lower probability = higher anomaly score
        threshold = np.percentile(log_probs, 10)  # Bottom 10%
        anomaly_scores = (threshold - log_probs) * 10
        
        # Clip to reasonable range
        anomaly_scores = np.clip(anomaly_scores, -100, 100)
        
        return pd.Series(anomaly_scores, index=features.index)


class MultivarAnomalySignal(MLSignalBase):
    """Multivariate anomaly detection combining multiple factors."""
    
    def __init__(self, config: SignalConfig):
        super().__init__(config)
    
    @property
    def name(self) -> str:
        return "MultivarAnomaly"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_ANOMALY
    
    @property
    def description(self) -> str:
        return "Multivariate anomaly detection"
    
    @property
    def dependencies(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive multivariate features."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data['high'] 
        low = data['low']
        volume = data['volume']
        returns = close.pct_change()
        
        # Market microstructure
        features['spread'] = (high - low) / close
        features['close_location'] = (close - low) / (high - low + 0.0001)
        features['volume_price_corr'] = returns.rolling(20).corr(volume.pct_change())
        
        # Multi-timeframe features
        for tf in [5, 20, 60]:
            features[f'return_{tf}'] = close.pct_change(tf)
            features[f'volume_ratio_{tf}'] = volume / volume.rolling(tf).mean()
            features[f'volatility_{tf}'] = returns.rolling(tf).std() * np.sqrt(252)
        
        # Cross-sectional features
        features['price_volume_divergence'] = (
            close.pct_change().rolling(20).mean() - 
            volume.pct_change().rolling(20).mean()
        )
        
        # Efficiency and liquidity
        features['amihud_illiquidity'] = returns.abs() / (volume * close)
        features['roll_spread'] = 2 * np.sqrt(abs(
            returns.rolling(20).cov(returns.shift(1))
        ))
        
        # Tail behavior
        features['left_tail'] = returns.rolling(50).quantile(0.05)
        features['right_tail'] = returns.rolling(50).quantile(0.95)
        features['tail_ratio'] = features['right_tail'] / abs(features['left_tail'])
        
        # Information flow
        features['info_ratio'] = returns.rolling(20).mean() / returns.rolling(20).std()
        
        # Complexity measures
        features['return_entropy'] = returns.rolling(20).apply(
            lambda x: -np.sum(np.histogram(x, bins=10)[0] * 
                            np.log(np.histogram(x, bins=10)[0] + 1e-10))
        )
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """No labels for anomaly detection."""
        return pd.Series(index=data.index)
    
    def create_model(self) -> Any:
        """Create Isolation Forest for multivariate anomaly detection."""
        return IsolationForest(
            n_estimators=200,
            contamination=0.05,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )