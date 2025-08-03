"""Ensemble ML signals combining multiple models."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
import xgboost as xgb

from .base import MLSignalBase
from ..base import SignalType
from ..base.config import SignalConfig


class EnsembleDirectionSignal(MLSignalBase):
    """Ensemble model for price direction prediction."""
    
    def __init__(self, config: SignalConfig,
                 prediction_horizon: Optional[int] = None):
        super().__init__(config, prediction_horizon=prediction_horizon)
    
    @property
    def name(self) -> str:
        return f"EnsembleDirection_{self.prediction_horizon}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_CLASSIFICATION
    
    @property
    def description(self) -> str:
        return f"Ensemble direction classifier with {self.prediction_horizon}-period horizon"
    
    @property
    def dependencies(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for ensemble."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        returns = close.pct_change()
        
        # Technical features
        for period in [5, 10, 20, 50]:
            # Price features
            features[f'return_{period}'] = close.pct_change(period)
            features[f'sma_ratio_{period}'] = close / close.rolling(period).mean() - 1
            features[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
            
            # Volume features
            features[f'volume_ratio_{period}'] = volume / volume.rolling(period).mean()
        
        # Momentum indicators
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        features['rsi'] = 100 - (100 / (1 + gain / loss))
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        features['macd'] = (ema12 - ema26) / close
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # Stochastic
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        features['stochastic'] = 100 * (close - low_14) / (high_14 - low_14)
        
        # Market structure
        features['high_low_ratio'] = (high - low) / close
        features['close_location'] = (close - low) / (high - low + 0.0001)
        
        # Trend strength
        for period in [20, 50]:
            features[f'trend_strength_{period}'] = close.rolling(period).apply(
                lambda x: abs(np.polyfit(range(len(x)), x, 1)[0]) / x.std()
            )
        
        # Statistical features
        features['skewness'] = returns.rolling(20).skew()
        features['kurtosis'] = returns.rolling(20).kurt()
        
        # Microstructure
        features['efficiency_ratio'] = close.diff(20).abs() / \
                                      close.diff().abs().rolling(20).sum()
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create labels: -1 (down), 0 (neutral), 1 (up)."""
        close = data['close']
        future_returns = close.shift(-self.prediction_horizon) / close - 1
        
        labels = pd.Series(0, index=data.index)
        labels[future_returns > 0.001] = 1
        labels[future_returns < -0.001] = -1
        
        return labels
    
    def create_model(self) -> Any:
        """Create voting ensemble of multiple classifiers."""
        # Define base models
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        svm = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('xgb', xgb_clf),
                ('svm', svm)
            ],
            voting='soft',  # Use probability voting
            weights=[2, 2, 3, 1]  # Weight XGBoost higher
        )
        
        return ensemble
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return ensemble probability scores."""
        if not self.is_trained or self.model is None:
            raise ValueError(f"{self.name} model not trained")
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get probability predictions
        probas = self.model.predict_proba(features_scaled)
        
        # Calculate directional score
        if -1 in self.model.classes_:
            down_idx = list(self.model.classes_).index(-1)
            up_idx = list(self.model.classes_).index(1)
            score = probas[:, up_idx] - probas[:, down_idx]
        else:
            score = probas[:, -1] - probas[:, 0]  # Assume ordered classes
        
        return pd.Series(score * 100, index=features.index)


class StackedMLSignal(MLSignalBase):
    """Stacked model combining multiple ML approaches."""
    
    def __init__(self, config: SignalConfig):
        super().__init__(config)
        self.base_models = {}
        self.meta_model = None
    
    @property
    def name(self) -> str:
        return "StackedML"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_REGRESSION
    
    @property
    def description(self) -> str:
        return "Stacked ML model with meta-learning"
    
    @property
    def dependencies(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create diverse features for stacking."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()
        
        # Group 1: Short-term features
        for period in [3, 5, 10]:
            features[f'st_return_{period}'] = returns.rolling(period).mean()
            features[f'st_volatility_{period}'] = returns.rolling(period).std()
            features[f'st_volume_{period}'] = volume / volume.rolling(period).mean()
        
        # Group 2: Medium-term features
        for period in [20, 30]:
            features[f'mt_return_{period}'] = returns.rolling(period).mean()
            features[f'mt_volatility_{period}'] = returns.rolling(period).std()
            features[f'mt_trend_{period}'] = close / close.rolling(period).mean() - 1
        
        # Group 3: Long-term features
        for period in [50, 100]:
            features[f'lt_return_{period}'] = returns.rolling(period).mean()
            features[f'lt_volatility_{period}'] = returns.rolling(period).std()
            features[f'lt_trend_{period}'] = close / close.rolling(period).mean() - 1
        
        # Group 4: Pattern features
        features['pattern_momentum'] = close / close.shift(10) - 1
        features['pattern_mean_reversion'] = (close - close.rolling(50).mean()) / \
                                            close.rolling(50).std()
        features['pattern_breakout'] = (close - close.rolling(20).max()) / close
        
        # Group 5: Statistical features
        features['stat_skew'] = returns.rolling(30).skew()
        features['stat_kurt'] = returns.rolling(30).kurt()
        features['stat_autocorr'] = returns.rolling(30).apply(lambda x: x.autocorr())
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create labels: future returns."""
        close = data['close']
        return (close.shift(-self.prediction_horizon) / close - 1) * 100
    
    def create_model(self) -> Any:
        """Create stacking regressor."""
        # Base models
        rf_reg = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        
        gb_reg = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        
        xgb_reg = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=6,
            random_state=42
        )
        
        # Meta learner
        meta_learner = Ridge(alpha=1.0)
        
        # Create stacking regressor
        stacking = StackingRegressor(
            estimators=[
                ('rf', rf_reg),
                ('gb', gb_reg),
                ('xgb', xgb_reg)
            ],
            final_estimator=meta_learner,
            cv=3  # 3-fold cross-validation for training meta-learner
        )
        
        return stacking


class VotingClassifierSignal(MLSignalBase):
    """Voting classifier for regime detection."""
    
    def __init__(self, config: SignalConfig,
                 n_regimes: int = 3):
        super().__init__(config)
        self.n_regimes = n_regimes
    
    @property
    def name(self) -> str:
        return f"VotingRegime_{self.n_regimes}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.ML_CLASSIFICATION
    
    @property
    def description(self) -> str:
        return f"Voting classifier for {self.n_regimes} market regimes"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close', 'volume']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create regime-specific features."""
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()
        
        # Volatility regime features
        for period in [10, 20, 50]:
            vol = returns.rolling(period).std() * np.sqrt(252)
            features[f'volatility_{period}'] = vol
            features[f'vol_percentile_{period}'] = vol.rolling(252).rank(pct=True)
        
        # Trend regime features
        for period in [20, 50, 100]:
            trend = close / close.rolling(period).mean() - 1
            features[f'trend_{period}'] = trend
            features[f'trend_strength_{period}'] = close.rolling(period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.std()
            )
        
        # Volume regime features
        features['volume_regime'] = volume / volume.rolling(50).mean()
        features['volume_trend'] = volume.rolling(20).mean() / volume.rolling(50).mean()
        
        # Market breadth
        features['price_dispersion'] = returns.rolling(20).std() / returns.rolling(20).mean().abs()
        features['efficiency_ratio'] = close.diff(20).abs() / close.diff().abs().rolling(20).sum()
        
        # Correlation structure
        features['autocorr_5'] = returns.rolling(50).apply(lambda x: x.autocorr(lag=5))
        features['autocorr_10'] = returns.rolling(50).apply(lambda x: x.autocorr(lag=10))
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create regime labels based on volatility and trend."""
        close = data['close']
        returns = close.pct_change()
        
        # Calculate metrics
        volatility = returns.rolling(20).std() * np.sqrt(252)
        trend = close.rolling(50).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Normalize
        vol_norm = (volatility - volatility.rolling(252).mean()) / volatility.rolling(252).std()
        trend_norm = (trend - trend.rolling(252).mean()) / trend.rolling(252).std()
        
        # Define regimes
        labels = pd.Series(0, index=data.index)
        
        if self.n_regimes == 3:
            # Bull, Bear, Sideways
            labels[trend_norm > 0.5] = 0  # Bull
            labels[trend_norm < -0.5] = 1  # Bear
            labels[(trend_norm >= -0.5) & (trend_norm <= 0.5)] = 2  # Sideways
        elif self.n_regimes == 4:
            # Add volatile regime
            labels[(vol_norm > 1) & (trend_norm > 0)] = 0  # Volatile Bull
            labels[(vol_norm > 1) & (trend_norm <= 0)] = 1  # Volatile Bear
            labels[(vol_norm <= 1) & (trend_norm > 0)] = 2  # Quiet Bull
            labels[(vol_norm <= 1) & (trend_norm <= 0)] = 3  # Quiet Bear
        
        return labels
    
    def create_model(self) -> Any:
        """Create diverse voting classifier."""
        # Use different algorithms for diversity
        rf = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            random_state=42
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=100,
            loss='log_loss',
            random_state=43
        )
        
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            objective='multi:softprob',
            random_state=44
        )
        
        # Logistic regression for linear patterns
        lr = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            random_state=45
        )
        
        # Create voting classifier
        voting = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('xgb', xgb_clf),
                ('lr', lr)
            ],
            voting='soft'
        )
        
        return voting
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return regime probabilities as signal."""
        if not self.is_trained or self.model is None:
            raise ValueError(f"{self.name} model not trained")
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get probability predictions
        probas = self.model.predict_proba(features_scaled)
        
        # Convert to signal based on regime probabilities
        # Example mapping for 3 regimes: Bull=1, Bear=-1, Sideways=0
        if self.n_regimes == 3:
            signal = probas[:, 0] - probas[:, 1]  # Bull - Bear
        else:
            # For 4 regimes, combine bull regimes vs bear regimes
            bull_prob = probas[:, 0] + probas[:, 2]  # Volatile + Quiet Bull
            bear_prob = probas[:, 1] + probas[:, 3]  # Volatile + Quiet Bear
            signal = bull_prob - bear_prob
        
        return pd.Series(signal * 100, index=features.index)