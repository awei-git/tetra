"""Clustering-based ML signals."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import logging

from .base import MLSignalBase
from ..base import SignalType
from ..base.config import SignalConfig

logger = logging.getLogger(__name__)


class MarketStateClusteringSignal(MLSignalBase):
    """Clusters market states based on multiple indicators."""
    
    def __init__(self, config: SignalConfig,
                 n_clusters: int = 5):
        super().__init__(config)
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
    
    @property
    def name(self) -> str:
        return f"MarketStateClustering_{self.n_clusters}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_CLUSTERING
    
    @property
    def description(self) -> str:
        return f"Market state clustering with {self.n_clusters} clusters"
    
    @property
    def dependencies(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create market state features."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        returns = close.pct_change()
        
        # Trend indicators
        for period in [10, 20, 50]:
            sma = close.rolling(period).mean()
            features[f'trend_{period}'] = (close - sma) / sma
            features[f'trend_strength_{period}'] = close.rolling(period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.std()
            )
        
        # Volatility indicators
        for period in [10, 20]:
            features[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
            features[f'volatility_ratio_{period}'] = features[f'volatility_{period}'] / \
                                                    features[f'volatility_{period}'].rolling(50).mean()
        
        # Volume indicators
        features['volume_ratio'] = volume / volume.rolling(20).mean()
        features['volume_trend'] = volume.rolling(10).mean() / volume.rolling(50).mean()
        
        # Market breadth
        features['high_low_spread'] = (high.rolling(20).max() - low.rolling(20).min()) / close
        features['close_position'] = (close - low.rolling(20).min()) / \
                                    (high.rolling(20).max() - low.rolling(20).min())
        
        # Momentum
        features['rsi'] = self._calculate_rsi(close, 14)
        features['momentum_10'] = close / close.shift(10) - 1
        features['momentum_20'] = close / close.shift(20) - 1
        
        # Market efficiency
        features['efficiency_ratio'] = close.diff(20).abs() / \
                                      close.diff().abs().rolling(20).sum()
        
        return features.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """No labels for clustering."""
        return pd.Series(index=data.index)
    
    def create_model(self) -> Any:
        """Create KMeans clustering model."""
        return KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
    
    def train(self, features: pd.DataFrame, labels: pd.Series):
        """Override train for clustering."""
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
        
        # Store cluster centers
        self.cluster_centers_ = self.model.cluster_centers_
        
        # Calculate training metrics
        cluster_labels = self.model.labels_
        cluster_counts = pd.Series(cluster_labels).value_counts()
        
        self.train_metrics = {
            'train_samples': len(features),
            'inertia': self.model.inertia_,
            'n_clusters': self.n_clusters,
            'cluster_sizes': cluster_counts.to_dict()
        }
        
        self.is_trained = True
        self.last_train_date = features.index[-1]
        
        logger.info(f"{self.name} trained with {len(features)} samples")
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return cluster assignments and distances."""
        if not self.is_trained or self.model is None:
            raise ValueError(f"{self.name} model not trained")
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get cluster assignments
        clusters = self.model.predict(features_scaled)
        
        # Calculate distance to nearest cluster center
        distances = self.model.transform(features_scaled)
        min_distances = distances.min(axis=1)
        
        # Normalize distances
        mean_dist = min_distances.mean()
        std_dist = min_distances.std()
        normalized_distances = (min_distances - mean_dist) / std_dist
        
        # Combine cluster assignment and distance
        # Positive values indicate unusual states (far from any cluster)
        result = clusters + normalized_distances * 0.1
        
        return pd.Series(result * 10, index=features.index)  # Scale to reasonable range


class PriceActionClusteringSignal(MLSignalBase):
    """Clusters price action patterns."""
    
    def __init__(self, config: SignalConfig,
                 pattern_length: int = 20,
                 n_clusters: int = 8):
        super().__init__(config, feature_window=pattern_length)
        self.pattern_length = pattern_length
        self.n_clusters = n_clusters
    
    @property
    def name(self) -> str:
        return f"PriceActionClustering_{self.n_clusters}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_CLUSTERING
    
    @property
    def description(self) -> str:
        return f"Price action pattern clustering with {self.n_clusters} patterns"
    
    @property
    def dependencies(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create normalized price patterns."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Create normalized price patterns
        for i in range(self.pattern_length):
            # Normalized prices
            features[f'close_{i}'] = close.shift(i) / close - 1
            features[f'high_{i}'] = high.shift(i) / close - 1
            features[f'low_{i}'] = low.shift(i) / close - 1
            
            # Normalized volume
            features[f'volume_{i}'] = volume.shift(i) / volume.rolling(50).mean()
        
        # Pattern shape features
        pattern_closes = pd.DataFrame({
            f'c_{i}': close.shift(i) for i in range(self.pattern_length)
        })
        
        # Trend within pattern
        features['pattern_trend'] = pattern_closes.apply(
            lambda row: np.polyfit(range(len(row)), row.values, 1)[0] / row.mean()
            if not row.isna().any() else np.nan,
            axis=1
        )
        
        # Volatility within pattern
        features['pattern_volatility'] = pattern_closes.std(axis=1) / pattern_closes.mean(axis=1)
        
        # Pattern smoothness
        features['pattern_smoothness'] = pattern_closes.diff(axis=1).abs().mean(axis=1) / \
                                        pattern_closes.mean(axis=1)
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """No labels for clustering."""
        return pd.Series(index=data.index)
    
    def create_model(self) -> Any:
        """Create Gaussian Mixture model for soft clustering."""
        return GaussianMixture(
            n_components=self.n_clusters,
            covariance_type='diag',
            random_state=42
        )
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return pattern probabilities."""
        if not self.is_trained or self.model is None:
            raise ValueError(f"{self.name} model not trained")
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get probabilities for each pattern
        probabilities = self.model.predict_proba(features_scaled)
        
        # Find most likely pattern and its probability
        max_prob_idx = probabilities.argmax(axis=1)
        max_prob = probabilities.max(axis=1)
        
        # Create signal based on pattern and confidence
        # Patterns 0-3: bearish, 4-7: bullish (example)
        signal = np.where(max_prob_idx < self.n_clusters // 2, -1, 1)
        signal = signal * max_prob * 100
        
        return pd.Series(signal, index=features.index)


class VolumeProfileClusteringSignal(MLSignalBase):
    """Clusters volume profiles to identify accumulation/distribution."""
    
    def __init__(self, config: SignalConfig,
                 profile_window: int = 20,
                 n_clusters: int = 4):
        super().__init__(config, feature_window=profile_window)
        self.profile_window = profile_window
        self.n_clusters = n_clusters
    
    @property
    def name(self) -> str:
        return f"VolumeProfileClustering_{self.n_clusters}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_CLUSTERING
    
    @property
    def description(self) -> str:
        return f"Volume profile clustering with {self.n_clusters} profiles"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volume profile features."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        returns = close.pct_change()
        
        # Volume distribution features
        for i in range(0, self.profile_window, 4):  # Sample every 4 periods
            features[f'volume_profile_{i}'] = volume.shift(i) / volume.rolling(self.profile_window).sum()
            features[f'price_level_{i}'] = close.shift(i) / close
        
        # Volume-price relationship
        features['volume_price_corr'] = returns.rolling(self.profile_window).corr(
            volume.pct_change()
        )
        
        # Volume concentration
        features['volume_concentration'] = volume.rolling(5).sum() / \
                                          volume.rolling(self.profile_window).sum()
        
        # Price range during volume
        features['price_range'] = (high.rolling(self.profile_window).max() - 
                                  low.rolling(self.profile_window).min()) / close
        
        # Volume-weighted price
        vwap = (close * volume).rolling(self.profile_window).sum() / \
               volume.rolling(self.profile_window).sum()
        features['vwap_deviation'] = (close - vwap) / vwap
        
        # Volume momentum
        features['volume_momentum'] = volume.rolling(5).mean() / \
                                     volume.rolling(self.profile_window).mean()
        
        # Accumulation/Distribution indicators
        money_flow_mult = ((close - low) - (high - close)) / (high - low)
        money_flow_volume = money_flow_mult * volume
        features['adl_slope'] = money_flow_volume.rolling(self.profile_window).sum().diff(5)
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """No labels for clustering."""
        return pd.Series(index=data.index)
    
    def create_model(self) -> Any:
        """Create Agglomerative clustering for hierarchical patterns."""
        return AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage='ward'
        )
    
    def train(self, features: pd.DataFrame, labels: pd.Series):
        """Override train for AgglomerativeClustering."""
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        if len(features) < self.config.ml_min_train_samples:
            raise ValueError(f"Insufficient training samples: {len(features)}")
        
        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Create and train model
        self.model = self.create_model()
        cluster_labels = self.model.fit_predict(features_scaled)
        
        # Store cluster assignments for later use
        self._train_features = features_scaled
        self._train_labels = cluster_labels
        
        # Calculate training metrics
        cluster_counts = pd.Series(cluster_labels).value_counts()
        
        self.train_metrics = {
            'train_samples': len(features),
            'n_clusters': self.n_clusters,
            'cluster_sizes': cluster_counts.to_dict()
        }
        
        self.is_trained = True
        self.last_train_date = features.index[-1]
        
        logger.info(f"{self.name} trained with {len(features)} samples")
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Assign to nearest cluster based on training data."""
        if not self.is_trained:
            raise ValueError(f"{self.name} model not trained")
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # For each new point, find nearest cluster
        results = []
        
        for i in range(len(features_scaled)):
            # Calculate distances to all training points
            distances = np.sqrt(np.sum((self._train_features - features_scaled[i])**2, axis=1))
            
            # Find k nearest neighbors
            k = min(10, len(self._train_features))
            nearest_idx = np.argpartition(distances, k)[:k]
            
            # Get their cluster labels
            nearest_clusters = self._train_labels[nearest_idx]
            
            # Vote for cluster
            cluster_counts = pd.Series(nearest_clusters).value_counts()
            assigned_cluster = cluster_counts.index[0]
            confidence = cluster_counts.iloc[0] / k
            
            # Map clusters to signals
            # 0: Strong accumulation, 1: Accumulation, 2: Distribution, 3: Strong distribution
            cluster_signals = {0: 50, 1: 25, 2: -25, 3: -50}
            signal = cluster_signals.get(assigned_cluster, 0) * confidence
            
            results.append(signal)
        
        return pd.Series(results, index=features.index)