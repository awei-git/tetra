"""ML metrics calculator for the metrics pipeline."""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    ExtraTreesRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings('ignore')

# Import ML definitions
from src.definitions.ml import (
    ML_MODEL_CONFIGS, MLFeatureConfig, MLTradingConfig, 
    MLMetricsConfig, MLEnsembleConfig, MLOutputConfig
)

# Import time series specific models
try:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    HAS_BOOSTING = True
except ImportError:
    HAS_BOOSTING = False
    
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

logger = logging.getLogger(__name__)


class MLMetricsCalculator:
    """
    Calculates ML predictions and metrics for ML-based trading strategies.
    This runs in the metrics pipeline to pre-compute ML signals that strategies can use.
    """
    
    def __init__(self):
        """Initialize ML metrics calculator with models from definitions."""
        
        # Load configurations
        self.feature_config = MLFeatureConfig()
        self.trading_config = MLTradingConfig()
        self.metrics_config = MLMetricsConfig()
        self.ensemble_config = MLEnsembleConfig()
        self.output_config = MLOutputConfig()
        
        # Initialize models from definitions
        self.models = {}
        for name, config in ML_MODEL_CONFIGS.items():
            params = config.parameters
            
            if name == 'rf_regressor':
                self.models[name] = RandomForestRegressor(**params)
            elif name == 'extra_trees':
                self.models[name] = ExtraTreesRegressor(**params)
            elif name == 'gb_regressor':
                self.models[name] = GradientBoostingRegressor(**params)
            elif name == 'hist_gb':
                self.models[name] = HistGradientBoostingRegressor(**params)
            elif name == 'ada_boost':
                self.models[name] = AdaBoostRegressor(**params)
            elif name == 'mlp':
                self.models[name] = MLPRegressor(**params)
            elif name == 'svr':
                self.models[name] = SVR(**params)
            elif name == 'ridge':
                self.models[name] = Ridge(**params)
            elif name == 'elastic_net':
                self.models[name] = ElasticNet(**params)
            elif HAS_BOOSTING:
                if name == 'xgboost':
                    self.models[name] = XGBRegressor(**params)
                elif name == 'lightgbm':
                    self.models[name] = LGBMRegressor(**params)
                elif name == 'catboost':
                    self.models[name] = CatBoostRegressor(**params)
        
        # Time series specific models
        self.time_series_models = {}
        if HAS_STATSMODELS:
            # These will be initialized per symbol since they need specific data
            self.time_series_models = {
                'arima': None,  # Will be fitted per series
                'sarimax': None,  # Seasonal ARIMA
                'exp_smoothing': None,  # Exponential smoothing
            }
        
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.model_weights = {}  # For weighted ensemble
        
    def calculate_ml_metrics(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Calculate ML predictions and metrics for the given data.
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            symbol: Trading symbol
            
        Returns:
            DataFrame with ML predictions and metrics
        """
        min_samples = ML_MODEL_CONFIGS['rf_regressor'].min_training_samples
        if data.empty or len(data) < min_samples:
            logger.warning(f"Insufficient data for {symbol}: {len(data)} rows, need {min_samples}")
            return pd.DataFrame()
        
        try:
            # Prepare features and targets
            X, y_regression, y_classification = self._prepare_ml_features(data)
            
            if X.empty or len(X) < 100:
                logger.warning(f"Insufficient features for {symbol}")
                return pd.DataFrame()
            
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=self.metrics_config.cv_splits)
            
            # Initialize result DataFrame
            ml_metrics = pd.DataFrame(index=data.index)
            
            # Train and predict with each model
            for model_name, model in self.models.items():
                logger.info(f"Training {model_name} for {symbol}")
                
                try:
                    # Scale features
                    X_scaled = self.scaler.fit_transform(X)
                    
                    # For time series cross-validation
                    cv_scores = []
                    for train_idx, val_idx in tscv.split(X):
                        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                        y_train, y_val = y_regression.iloc[train_idx], y_regression.iloc[val_idx]
                        
                        # Fit model on train fold
                        model_copy = self._clone_model(model)
                        model_copy.fit(X_train, y_train)
                        
                        # Validate
                        val_pred = model_copy.predict(X_val)
                        cv_scores.append(mean_squared_error(y_val, val_pred))
                    
                    # Store CV score for model weighting
                    self.model_weights[model_name] = 1 / (np.mean(cv_scores) + 1e-10)
                    
                    # Final training on all data except test portion
                    train_size = int(len(X) * (1 - self.metrics_config.test_size))
                    X_train = X_scaled[:train_size]
                    y_train = y_regression.iloc[:train_size]
                    
                    # Fit final model
                    model.fit(X_train, y_train)
                    
                    # Make predictions - ensure they align with data index
                    predictions = model.predict(X_scaled)
                    # Create a Series with the same index as the original data
                    pred_series = pd.Series(predictions, index=data.index)
                    ml_metrics[f'{model_name}_prediction'] = pred_series
                    
                    # Calculate prediction confidence
                    if hasattr(model, 'estimators_'):
                        # For ensemble models, use prediction variance
                        tree_predictions = np.array([
                            estimator.predict(X_scaled) 
                            for estimator in model.estimators_[:min(10, len(model.estimators_))]
                        ])
                        confidence = 1 / (1 + tree_predictions.std(axis=0))
                        ml_metrics[f'{model_name}_confidence'] = pd.Series(confidence, index=data.index)
                    else:
                        # For other models, use distance from mean prediction
                        pred_std = np.abs(predictions - predictions.mean())
                        confidence = 1 / (1 + pred_std)
                        ml_metrics[f'{model_name}_confidence'] = pd.Series(confidence, index=data.index)
                    
                    # Store feature importance
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[model_name] = dict(zip(X.columns, model.feature_importances_))
                        
                except Exception as e:
                    logger.warning(f"Failed to train {model_name}: {e}")
                    ml_metrics[f'{model_name}_prediction'] = 0
                    ml_metrics[f'{model_name}_confidence'] = 0
            
            # Add time series specific predictions if available
            if HAS_STATSMODELS:
                ts_predictions = self._calculate_time_series_predictions(data)
                for ts_name, ts_pred in ts_predictions.items():
                    ml_metrics[f'{ts_name}_prediction'] = ts_pred
                    ml_metrics[f'{ts_name}_confidence'] = 0.5  # Default confidence for TS models
            
            # Calculate weighted ensemble predictions
            ml_metrics['ml_ensemble_prediction'] = self._calculate_weighted_ensemble(ml_metrics)
            
            # Generate advanced trading signals
            ml_metrics['ml_signal'] = self._generate_ml_signals(ml_metrics)
            ml_metrics['ml_signal_strength'] = self._calculate_signal_strength(ml_metrics)
            
            # Calculate ML strategy metrics
            ml_metrics['ml_expected_return'] = ml_metrics['ml_ensemble_prediction']
            ml_metrics['ml_risk_score'] = self._calculate_ml_risk_score(ml_metrics, data)
            
            # Add classification predictions
            ml_metrics['ml_action'] = self._classify_action(ml_metrics)
            ml_metrics['ml_position_size'] = self._calculate_position_size(ml_metrics)
            
            # Calculate model performance metrics on test set
            if train_size < len(X) - 20:
                test_metrics = self._calculate_model_performance(
                    X_scaled[train_size:],
                    y_regression.iloc[train_size:],
                    y_classification.iloc[train_size:],
                    ml_metrics.iloc[train_size:]
                )
                
                # Add performance metrics
                for metric_name, value in test_metrics.items():
                    ml_metrics[f'ml_{metric_name}'] = value
            
            # Add prediction intervals (uncertainty quantification)
            ml_metrics['ml_prediction_lower'] = ml_metrics['ml_ensemble_prediction'] - ml_metrics['ml_risk_score'] * 0.02
            ml_metrics['ml_prediction_upper'] = ml_metrics['ml_ensemble_prediction'] + ml_metrics['ml_risk_score'] * 0.02
            
            # Add feature importance as JSON
            ml_metrics['ml_feature_importance'] = str(self.feature_importance)
            
            # Add anomaly detection scores
            anomaly_scores = self._calculate_anomaly_scores(data, ml_metrics)
            ml_metrics['ml_anomaly_score'] = anomaly_scores
            
            # Save trained models for ML strategies to use
            self._save_models(symbol)
            
            return ml_metrics
            
        except Exception as e:
            logger.error(f"Error calculating ML metrics for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare features and targets for ML models with time series specific features.
        """
        features = []
        
        # Price-based features (returns at multiple horizons)
        if 'close' in data.columns:
            for lag in [1, 2, 3, 5, 10, 20, 60]:
                features.append(data['close'].pct_change(lag).fillna(0).rename(f'return_{lag}d'))
            
            # Log returns for better statistical properties
            features.append(np.log(data['close'] / data['close'].shift(1)).fillna(0).rename('log_return'))
            
            # Price relative to moving averages
            for ma in [5, 10, 20, 50, 100, 200]:
                if f'sma_{ma}' in data.columns:
                    features.append((data['close'] / data[f'sma_{ma}'] - 1).rename(f'close_to_sma{ma}'))
            
            # Price momentum
            features.append((data['close'] / data['close'].shift(20) - 1).fillna(0).rename('momentum_20d'))
            features.append((data['close'] / data['close'].shift(60) - 1).fillna(0).rename('momentum_60d'))
        
        # Volume features
        if 'volume' in data.columns:
            features.append((data['volume'] / data['volume'].rolling(20).mean() - 1).fillna(0).rename('volume_ratio'))
            features.append(data['volume'].pct_change(1).fillna(0).rename('volume_change'))
            
            # Volume-weighted features
            if 'vwap' in data.columns:
                features.append((data['close'] / data['vwap'] - 1).fillna(0).rename('close_to_vwap'))
        
        # Technical indicators
        technical_features = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_middle',
            'atr', 'adx', 'cci', 'mfi', 'roc', 'williams_r',
            'stochastic_k', 'stochastic_d', 'obv_normalized'
        ]
        for feat in technical_features:
            if feat in data.columns:
                features.append(data[feat].fillna(method='ffill').fillna(0).rename(feat))
        
        # Statistical features
        if 'close' in data.columns:
            # Rolling statistics
            for window in [5, 10, 20, 60]:
                returns = data['close'].pct_change()
                features.append(returns.rolling(window).mean().fillna(0).rename(f'mean_return_{window}d'))
                features.append(returns.rolling(window).std().fillna(0).rename(f'volatility_{window}d'))
                features.append(returns.rolling(window).skew().fillna(0).rename(f'skewness_{window}d'))
                features.append(returns.rolling(window).kurt().fillna(0).rename(f'kurtosis_{window}d'))
        
        # Market microstructure features
        if 'high' in data.columns and 'low' in data.columns:
            features.append(((data['high'] - data['low']) / data['close']).fillna(0).rename('high_low_ratio'))
            features.append(((data['close'] - data['open']) / data['open']).fillna(0).rename('close_open_ratio'))
        
        # Lag features (autoregressive components)
        if 'close' in data.columns:
            returns = data['close'].pct_change()
            for lag in self.feature_config.ar_lags:
                features.append(returns.shift(lag).fillna(0).rename(f'return_lag_{lag}'))
        
        # Time-based features (seasonality)
        if isinstance(data.index, pd.DatetimeIndex):
            features.append(pd.Series(data.index.dayofweek, index=data.index, name='day_of_week'))
            features.append(pd.Series(data.index.day, index=data.index, name='day_of_month'))
            features.append(pd.Series(data.index.month, index=data.index, name='month'))
        
        # Market regime features
        if 'volatility_regime' in data.columns:
            # One-hot encode regime
            regime_dummies = pd.get_dummies(data['volatility_regime'], prefix='vol_regime')
            features.extend([regime_dummies[col] for col in regime_dummies.columns])
        
        if not features:
            return pd.DataFrame(), pd.Series(), pd.Series()
        
        # Combine features - ensure all have same index
        if features:
            # Ensure all features have the same index as data
            aligned_features = []
            for feature in features:
                if isinstance(feature, pd.Series):
                    feature = feature.reindex(data.index)
                aligned_features.append(feature)
            X = pd.concat(aligned_features, axis=1)
        else:
            X = pd.DataFrame(index=data.index)
        
        # Create targets for different prediction horizons
        # Primary target: next day return
        y_regression = data['close'].pct_change(1).shift(-1)
        
        # Multi-horizon targets for ensemble learning
        y_2d = data['close'].pct_change(2).shift(-2)
        y_5d = data['close'].pct_change(5).shift(-5)
        
        # Classification target
        y_classification = (y_regression > 0).astype(int)
        
        # Remove NaN rows
        valid_idx = X.notna().all(axis=1) & y_regression.notna()
        X = X[valid_idx]
        y_regression = y_regression[valid_idx]
        y_classification = y_classification[valid_idx]
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        return X, y_regression, y_classification
    
    def _calculate_time_series_predictions(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate predictions using time series specific models."""
        ts_predictions = {}
        
        if 'close' not in data.columns:
            return ts_predictions
        
        try:
            returns = data['close'].pct_change().dropna()
            
            # ARIMA model - use available data
            data_length = len(returns)
            arima_period = min(data_length, 252) if data_length > 60 else data_length
            
            try:
                if arima_period > 30:  # Need reasonable sample
                    arima = ARIMA(returns[-arima_period:], order=(2, 0, 2))  # ARMA(2,2) for returns
                    arima_fit = arima.fit()
                    # Forecast in-sample
                    arima_pred = arima_fit.fittedvalues
                    ts_predictions['arima'] = pd.Series(0, index=data.index)
                    ts_predictions['arima'].iloc[-len(arima_pred):] = arima_pred
            except:
                logger.warning("ARIMA fitting failed")
            
            # Exponential smoothing on prices - use available data
            try:
                if arima_period > 40:  # Need enough for seasonal
                    exp_smooth = ExponentialSmoothing(
                        data['close'][-arima_period:], 
                        trend='add', 
                        seasonal='add', 
                        seasonal_periods=min(20, arima_period // 2)
                    )
                    exp_fit = exp_smooth.fit()
                    exp_pred = exp_fit.fittedvalues.pct_change()
                    ts_predictions['exp_smoothing'] = pd.Series(0, index=data.index)
                    ts_predictions['exp_smoothing'].iloc[-len(exp_pred):] = exp_pred
            except:
                logger.warning("Exponential smoothing failed")
                
        except Exception as e:
            logger.warning(f"Time series prediction failed: {e}")
        
        return ts_predictions
    
    def _calculate_weighted_ensemble(self, ml_metrics: pd.DataFrame) -> pd.Series:
        """Calculate weighted ensemble prediction based on model performance."""
        prediction_cols = [col for col in ml_metrics.columns 
                         if 'prediction' in col and 'ensemble' not in col]
        
        if not prediction_cols:
            return pd.Series(0, index=ml_metrics.index)
        
        # Use model weights from cross-validation
        if self.model_weights:
            weighted_sum = pd.Series(0, index=ml_metrics.index)
            total_weight = 0
            
            for col in prediction_cols:
                model_name = col.replace('_prediction', '')
                weight = self.model_weights.get(model_name, 1.0)
                weighted_sum += ml_metrics[col] * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else weighted_sum
        else:
            # Simple average if no weights available
            return ml_metrics[prediction_cols].mean(axis=1)
    
    def _generate_ml_signals(self, ml_metrics: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on ML predictions."""
        signals = pd.Series(index=ml_metrics.index, dtype=str)
        
        # Use ensemble prediction for signals
        predictions = ml_metrics['ml_ensemble_prediction']
        confidence = ml_metrics[[col for col in ml_metrics.columns if 'confidence' in col]].mean(axis=1)
        
        # Dynamic thresholds based on prediction distribution
        pred_std = predictions.std()
        upper_threshold = predictions.quantile(0.7)
        lower_threshold = predictions.quantile(0.3)
        
        # Generate signals with dynamic thresholds
        signals[predictions > upper_threshold] = 'BUY'
        signals[predictions < lower_threshold] = 'SELL'
        signals[(predictions >= lower_threshold) & (predictions <= upper_threshold)] = 'HOLD'
        
        # Adjust for low confidence
        signals[confidence < 0.3] = 'WAIT'
        
        return signals.fillna('HOLD')
    
    def _calculate_signal_strength(self, ml_metrics: pd.DataFrame) -> pd.Series:
        """Calculate strength of ML signals (0-1 scale)."""
        predictions = ml_metrics['ml_ensemble_prediction'].abs()
        confidence = ml_metrics[[col for col in ml_metrics.columns if 'confidence' in col]].mean(axis=1)
        
        # Use percentile normalization for better scaling
        pred_normalized = predictions.rank(pct=True)
        
        # Combine prediction magnitude and confidence
        strength = (pred_normalized * 0.6 + confidence * 0.4)
        
        return strength.clip(0, 1)
    
    def _calculate_ml_risk_score(self, ml_metrics: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
        """Calculate risk score for ML predictions."""
        risk_score = pd.Series(index=ml_metrics.index, dtype=float)
        
        # Factor 1: Prediction disagreement across models
        prediction_cols = [col for col in ml_metrics.columns if 'prediction' in col and 'ensemble' not in col]
        if prediction_cols:
            prediction_std = ml_metrics[prediction_cols].std(axis=1)
            risk_score += prediction_std * 2
        
        # Factor 2: Market volatility
        if 'close' in data.columns:
            returns = data['close'].pct_change()
            rolling_vol = returns.rolling(20).std()
            risk_score += rolling_vol.reindex(ml_metrics.index).fillna(0.02) * 10
        
        # Factor 3: Prediction uncertainty (inverse confidence)
        confidence = ml_metrics[[col for col in ml_metrics.columns if 'confidence' in col]].mean(axis=1)
        risk_score += (1 - confidence)
        
        # Factor 4: Regime uncertainty
        if 'volatility_regime' in data.columns:
            regime_changes = (data['volatility_regime'] != data['volatility_regime'].shift()).astype(int)
            regime_uncertainty = regime_changes.rolling(20).sum() / 20
            risk_score += regime_uncertainty.reindex(ml_metrics.index).fillna(0)
        
        # Normalize to 0-1 scale
        risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min() + 1e-10)
        
        return risk_score.clip(0, 1)
    
    def _classify_action(self, ml_metrics: pd.DataFrame) -> pd.Series:
        """Classify trading action based on ML predictions."""
        actions = pd.Series(index=ml_metrics.index, dtype=str)
        
        predictions = ml_metrics['ml_ensemble_prediction']
        confidence = ml_metrics[[col for col in ml_metrics.columns if 'confidence' in col]].mean(axis=1)
        risk_score = ml_metrics['ml_risk_score']
        
        # Dynamic thresholds based on recent performance
        recent_predictions = predictions.rolling(20).mean()
        vol_adjusted_threshold = predictions.rolling(20).std() * 2
        
        # Strong signals (high confidence, low risk)
        strong_buy = (predictions > vol_adjusted_threshold) & (confidence > 0.7) & (risk_score < 0.3)
        strong_sell = (predictions < -vol_adjusted_threshold) & (confidence > 0.7) & (risk_score < 0.3)
        
        actions[strong_buy] = 'STRONG_BUY'
        actions[strong_sell] = 'STRONG_SELL'
        
        # Regular signals
        buy = (predictions > vol_adjusted_threshold * 0.5) & ~strong_buy
        sell = (predictions < -vol_adjusted_threshold * 0.5) & ~strong_sell
        
        actions[buy] = 'BUY'
        actions[sell] = 'SELL'
        
        # Hold zone
        hold = (predictions.abs() <= vol_adjusted_threshold * 0.5)
        actions[hold] = 'HOLD'
        
        # Wait for uncertain conditions
        actions[(confidence < 0.4) | (risk_score > 0.7)] = 'WAIT'
        
        return actions.fillna('HOLD')
    
    def _calculate_position_size(self, ml_metrics: pd.DataFrame) -> pd.Series:
        """Calculate recommended position size using Kelly Criterion variant."""
        predictions = ml_metrics['ml_ensemble_prediction']
        confidence = ml_metrics[[col for col in ml_metrics.columns if 'confidence' in col]].mean(axis=1)
        risk_score = ml_metrics['ml_risk_score']
        
        # Expected return (capped for stability)
        expected_return = predictions.clip(-0.1, 0.1)
        
        # Estimated probability of success based on confidence
        win_probability = 0.5 + confidence * 0.3  # Scale confidence to probability adjustment
        
        # Kelly fraction calculation (simplified)
        # f = (p * b - q) / b, where p = win prob, q = loss prob, b = win/loss ratio
        win_loss_ratio = 1.5  # Assume 1.5:1 reward/risk
        kelly_fraction = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
        
        # Apply fractional Kelly (25% of full Kelly for safety)
        position_size = kelly_fraction * 0.25
        
        # Adjust for risk
        position_size = position_size * (1 - risk_score * 0.5)
        
        # Adjust for prediction strength
        prediction_strength = predictions.abs() / (predictions.abs().quantile(0.95) + 1e-10)
        position_size = position_size * prediction_strength.clip(0, 1)
        
        # Cap at 20% max position
        return position_size.clip(0, 0.2)
    
    def _calculate_model_performance(self, X_test: np.ndarray, y_test_reg: pd.Series, 
                                    y_test_clf: pd.Series, predictions: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive model performance metrics."""
        metrics = {}
        
        # Regression metrics
        if 'ml_ensemble_prediction' in predictions.columns:
            y_pred = predictions['ml_ensemble_prediction'].values
            
            # Standard regression metrics
            metrics['mse'] = mean_squared_error(y_test_reg, y_pred)
            metrics['mae'] = mean_absolute_error(y_test_reg, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_test_reg, y_pred)
            
            # MAPE for percentage error
            mask = y_test_reg != 0
            if mask.any():
                metrics['mape'] = mean_absolute_percentage_error(y_test_reg[mask], y_pred[mask])
            
            # Directional accuracy
            y_pred_direction = (y_pred > 0).astype(int)
            y_true_direction = (y_test_reg > 0).astype(int)
            metrics['directional_accuracy'] = (y_pred_direction == y_true_direction).mean()
        
        # Classification metrics
        if 'ml_signal' in predictions.columns and len(np.unique(y_test_clf)) > 1:
            y_pred_direction = (predictions['ml_ensemble_prediction'] > 0).astype(int)
            
            metrics['accuracy'] = accuracy_score(y_test_clf, y_pred_direction)
            metrics['precision'] = precision_score(y_test_clf, y_pred_direction, zero_division=0)
            metrics['recall'] = recall_score(y_test_clf, y_pred_direction, zero_division=0)
            metrics['f1'] = f1_score(y_test_clf, y_pred_direction, zero_division=0)
            
            # ROC AUC
            if 'ml_ensemble_prediction' in predictions.columns:
                # Convert predictions to probabilities
                probs = 1 / (1 + np.exp(-predictions['ml_ensemble_prediction'].values * 100))
                try:
                    metrics['auc_roc'] = roc_auc_score(y_test_clf, probs)
                except:
                    metrics['auc_roc'] = 0.5
        
        # Trading specific metrics
        if 'ml_signal' in predictions.columns:
            trades = predictions['ml_signal'].isin(['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL'])
            if trades.any():
                trade_returns = y_test_reg[trades]
                trade_predictions = predictions['ml_ensemble_prediction'].iloc[-len(y_test_reg):][trades]
                
                # Hit rate (correct direction on trades)
                correct_trades = ((trade_predictions > 0) == (trade_returns > 0))
                metrics['hit_rate'] = correct_trades.mean()
                
                # Profit factor
                winning_trades = trade_returns[trade_returns > 0]
                losing_trades = trade_returns[trade_returns < 0]
                
                if len(winning_trades) > 0 and len(losing_trades) > 0:
                    metrics['profit_factor'] = winning_trades.sum() / abs(losing_trades.sum())
                elif len(winning_trades) > 0:
                    metrics['profit_factor'] = np.inf
                else:
                    metrics['profit_factor'] = 0
                
                # Average win/loss
                metrics['avg_win'] = winning_trades.mean() if len(winning_trades) > 0 else 0
                metrics['avg_loss'] = losing_trades.mean() if len(losing_trades) > 0 else 0
                
                # Sharpe ratio of predictions
                if len(trade_returns) > 1:
                    metrics['prediction_sharpe'] = (
                        trade_returns.mean() / trade_returns.std() 
                        if trade_returns.std() > 0 else 0
                    )
        
        return metrics
    
    def _calculate_anomaly_scores(self, data: pd.DataFrame, ml_metrics: pd.DataFrame) -> pd.Series:
        """
        Calculate anomaly scores using Isolation Forest and statistical methods.
        Anomalies are useful for detecting unusual market conditions or opportunities.
        """
        from sklearn.ensemble import IsolationForest
        
        anomaly_scores = pd.Series(index=ml_metrics.index, dtype=float).fillna(0)
        
        try:
            # Prepare features for anomaly detection
            features = []
            
            # Price and volume anomalies
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                # Z-score of returns
                returns_zscore = (returns - returns.rolling(20).mean()) / returns.rolling(20).std()
                features.append(returns_zscore.fillna(0))
                
                # Volume z-score
                if 'volume' in data.columns:
                    volume_zscore = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()
                    features.append(volume_zscore.fillna(0))
            
            # Technical indicator anomalies
            if 'rsi' in data.columns:
                # RSI extremes
                rsi_anomaly = ((data['rsi'] < 20) | (data['rsi'] > 80)).astype(float)
                features.append(rsi_anomaly)
            
            # Prediction anomalies
            if 'ml_ensemble_prediction' in ml_metrics.columns:
                pred_zscore = (ml_metrics['ml_ensemble_prediction'] - ml_metrics['ml_ensemble_prediction'].rolling(20).mean()) / ml_metrics['ml_ensemble_prediction'].rolling(20).std()
                features.append(pred_zscore.fillna(0))
            
            if features:
                # Combine features
                X = pd.concat(features, axis=1).fillna(0)
                
                # Isolation Forest for anomaly detection
                iso_forest = IsolationForest(
                    contamination=self.trading_config.anomaly_contamination,
                    random_state=42,
                    n_estimators=100
                )
                
                # Fit and predict
                anomaly_labels = iso_forest.fit_predict(X)
                anomaly_scores_raw = iso_forest.score_samples(X)
                
                # Convert to 0-1 scale (higher score = more anomalous)
                anomaly_scores = 1 - (anomaly_scores_raw - anomaly_scores_raw.min()) / (anomaly_scores_raw.max() - anomaly_scores_raw.min())
                
                # Boost scores for extreme anomalies
                anomaly_scores[anomaly_labels == -1] *= self.trading_config.anomaly_boost_factor
                anomaly_scores = anomaly_scores.clip(0, 1)
                
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            # Fallback to simple statistical anomaly
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                zscore = np.abs((returns - returns.mean()) / returns.std())
                anomaly_scores = (zscore / zscore.quantile(0.95)).clip(0, 1)
        
        return pd.Series(anomaly_scores, index=ml_metrics.index)
    
    def _save_models(self, symbol: str):
        """
        Save trained models to disk for ML strategies to use.
        Models are saved in a format compatible with MLPredictionStrategy.
        """
        import joblib
        from pathlib import Path
        
        try:
            # Create output directory
            output_dir = Path(self.output_config.models_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each model with metadata
            for model_name, model in self.models.items():
                if model is None:
                    continue
                    
                model_data = {
                    'model': model,
                    'symbol': symbol,
                    'feature_importance': self.feature_importance.get(model_name, {}),
                    'model_weight': self.model_weights.get(model_name, 1.0),
                    'scaler': self.scaler,
                    'timestamp': pd.Timestamp.now(),
                    'calibration_params': {
                        'prediction_threshold': self.trading_config.prediction_threshold,
                        'confidence_threshold': self.trading_config.normal_confidence,
                        'validation_metrics': {
                            'mae': 0.01,  # Will be updated with actual metrics
                            'mse': 0.0001,
                            'r2': 0.5
                        }
                    }
                }
                
                # Save model
                model_file = output_dir / f"{model_name}_{symbol}.pkl"
                joblib.dump(model_data, model_file)
                logger.debug(f"Saved model {model_name} for {symbol} to {model_file}")
                
            # Also save ensemble model metadata
            ensemble_data = {
                'models': list(self.models.keys()),
                'model_weights': self.model_weights,
                'feature_importance': self.feature_importance,
                'symbol': symbol,
                'timestamp': pd.Timestamp.now()
            }
            
            ensemble_file = output_dir / f"ensemble_{symbol}.pkl"
            joblib.dump(ensemble_data, ensemble_file)
            
        except Exception as e:
            logger.warning(f"Failed to save models for {symbol}: {e}")
    
    def _clone_model(self, model):
        """Clone a model for cross-validation."""
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            # For models that don't support sklearn clone
            return model.__class__(**model.get_params())
    
    @staticmethod
    def calculate_all(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Static method to calculate all ML metrics.
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            symbol: Trading symbol
            
        Returns:
            DataFrame with ML predictions and metrics
        """
        calculator = MLMetricsCalculator()
        return calculator.calculate_ml_metrics(data, symbol)