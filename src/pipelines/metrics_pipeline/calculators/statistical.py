"""Statistical metrics calculations for metrics pipeline."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class StatisticalCalculator:
    """Calculate statistical metrics for the metrics pipeline."""
    
    @staticmethod
    def calculate_returns(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate various return metrics."""
        metrics = {}
        data_length = len(df)
        
        # Simple returns for different periods - adapt to available data
        periods = [1, 5, 20, 60]
        if data_length >= 252:
            periods.append(252)
        
        for period in periods:
            if data_length >= period:
                metrics[f'returns_{period}'] = df['close'].pct_change(period)
                metrics[f'log_returns_{period}'] = np.log(df['close'] / df['close'].shift(period))
            else:
                # Create NaN series if not enough data
                metrics[f'returns_{period}'] = pd.Series(np.nan, index=df.index)
                metrics[f'log_returns_{period}'] = pd.Series(np.nan, index=df.index)
        
        # Cumulative returns
        metrics['cumulative_returns'] = (1 + df['close'].pct_change()).cumprod() - 1
        
        # Excess returns (simplified - would need risk-free rate)
        for period in [1, 5, 20]:
            risk_free_rate = 0.05 / 252  # Assuming 5% annual risk-free rate
            if f'returns_{period}' in metrics and not metrics[f'returns_{period}'].isna().all():
                metrics[f'excess_returns_{period}'] = metrics[f'returns_{period}'] - (risk_free_rate * period)
            else:
                metrics[f'excess_returns_{period}'] = pd.Series(np.nan, index=df.index)
        
        return metrics
    
    @staticmethod
    def calculate_volatility_metrics(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volatility metrics."""
        metrics = {}
        
        returns = df['close'].pct_change()
        
        # Realized volatility for different periods - adapt to available data
        data_length = len(df)
        vol_periods = [20]
        if data_length >= 60:
            vol_periods.append(60)
        if data_length >= 252:
            vol_periods.append(252)
            
        for period in vol_periods:
            if data_length >= period:
                metrics[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
        
        # EWMA volatility
        lambda_param = 0.94
        metrics['ewma_volatility'] = returns.ewm(alpha=1-lambda_param, adjust=False).std() * np.sqrt(252)
        
        # Parkinson volatility (using high-low)
        period = 20
        parkinson_factor = 1 / (4 * np.log(2))
        metrics['parkinson_volatility_20'] = np.sqrt(
            parkinson_factor * ((np.log(df['high'] / df['low']) ** 2).rolling(window=period).mean()) * 252
        )
        
        # Garman-Klass volatility
        garman_klass = (
            0.5 * (np.log(df['high'] / df['low']) ** 2) -
            (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2)
        )
        metrics['garman_klass_20'] = np.sqrt(garman_klass.rolling(window=period).mean() * 252)
        
        # Yang-Zhang volatility (simplified)
        metrics['yang_zhang_20'] = returns.rolling(window=period).std() * np.sqrt(252) * 1.34  # Adjustment factor
        
        return metrics
    
    @staticmethod
    def calculate_risk_metrics(df: pd.DataFrame, returns: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
        """Calculate risk metrics."""
        metrics = {}
        
        if returns is None:
            returns = df['close'].pct_change()
        
        # Value at Risk (VaR) - Historical method
        data_length = len(df)
        var_period = min(data_length - 1, 252) if data_length > 60 else min(data_length - 1, 60)
        
        if var_period > 20:  # Need reasonable sample size
            for confidence in [0.95, 0.99]:
                try:
                    # Use min_periods to avoid issues with NaN values
                    metrics[f'var_{int(confidence*100)}'] = returns.rolling(window=var_period, min_periods=int(var_period*0.5)).quantile(1 - confidence)
                except:
                    metrics[f'var_{int(confidence*100)}'] = pd.Series(np.nan, index=df.index)
            
            # Conditional VaR (CVaR) / Expected Shortfall
            for confidence in [0.95, 0.99]:
                try:
                    var = returns.rolling(window=var_period, min_periods=int(var_period*0.5)).quantile(1 - confidence)
                except:
                    var = pd.Series(np.nan, index=df.index)
                
                # Safe CVaR calculation
                def safe_cvar(x):
                    try:
                        if len(x.dropna()) == 0:
                            return np.nan
                        q = x.quantile(1 - confidence)
                        below_var = x[x <= q]
                        if len(below_var) > 0:
                            return below_var.mean()
                        return np.nan
                    except:
                        return np.nan
                
                metrics[f'cvar_{int(confidence*100)}'] = returns.rolling(window=var_period).apply(safe_cvar)
        
        # Maximum Drawdown - with safety for expanding operations
        try:
            cumulative = (1 + returns).cumprod()
            # Check if cumulative has valid data before expanding
            if not cumulative.isna().all() and len(cumulative.dropna()) > 0:
                # Use cummax() instead of expanding().max() to avoid argmax issues
                running_max = cumulative.cummax()
                drawdown = (cumulative - running_max) / running_max
                dd_period = min(len(df) - 1, 252) if len(df) > 60 else min(len(df) - 1, 60)
                if dd_period > 20:
                    metrics[f'max_drawdown_{dd_period}'] = drawdown.rolling(window=dd_period).min()
                
                # Ulcer Index
                if dd_period > 20:
                    drawdown_squared = drawdown ** 2
                    metrics[f'ulcer_index_{dd_period}'] = np.sqrt(drawdown_squared.rolling(window=dd_period).mean())
            else:
                # Create NaN series if no valid data
                dd_period = min(len(df) - 1, 252) if len(df) > 60 else min(len(df) - 1, 60)
                if dd_period > 20:
                    metrics[f'max_drawdown_{dd_period}'] = pd.Series(np.nan, index=df.index)
                    metrics[f'ulcer_index_{dd_period}'] = pd.Series(np.nan, index=df.index)
        except Exception as e:
            logger.warning(f"Error calculating drawdown metrics: {e}")
            dd_period = min(len(df) - 1, 252) if len(df) > 60 else min(len(df) - 1, 60)
            if dd_period > 20:
                metrics[f'max_drawdown_{dd_period}'] = pd.Series(np.nan, index=df.index)
                metrics[f'ulcer_index_{dd_period}'] = pd.Series(np.nan, index=df.index)
        
        # Downside Deviation
        try:
            threshold = 0
            downside_returns = returns.copy()
            downside_returns[downside_returns > threshold] = 0
            dd_period = min(len(df) - 1, 252) if len(df) > 60 else min(len(df) - 1, 60)
            if dd_period > 20:
                metrics[f'downside_deviation_{dd_period}'] = downside_returns.rolling(window=dd_period).std() * np.sqrt(252)
                # Semi-variance
                metrics[f'semi_variance_{dd_period}'] = (downside_returns ** 2).rolling(window=dd_period).mean()
        except Exception as e:
            logger.warning(f"Error calculating downside metrics: {e}")
            if dd_period > 20:
                metrics[f'downside_deviation_{dd_period}'] = pd.Series(np.nan, index=df.index)
                metrics[f'semi_variance_{dd_period}'] = pd.Series(np.nan, index=df.index)
        
        return metrics
    
    @staticmethod
    def calculate_correlation_metrics(df: pd.DataFrame, market_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.Series]:
        """Calculate correlation and beta metrics."""
        metrics = {}
        
        returns = df['close'].pct_change()
        
        # If market data provided, calculate correlations
        if market_df is not None and 'close' in market_df.columns:
            market_returns = market_df['close'].pct_change()
            
            # Rolling correlations - adapt to available data
            data_length = len(df)
            corr_periods = [20]
            if data_length >= 60:
                corr_periods.append(60)
            if data_length >= 252:
                corr_periods.append(252)
                
            for period in corr_periods:
                if data_length >= period:
                    metrics[f'correlation_spy_{period}'] = returns.rolling(window=period).corr(market_returns)
            
            # Beta
            beta_periods = []
            if data_length >= 60:
                beta_periods.append(60)
            if data_length >= 252:
                beta_periods.append(252)
                
            for period in beta_periods:
                if data_length >= period:
                    covariance = returns.rolling(window=period).cov(market_returns)
                    market_variance = market_returns.rolling(window=period).var()
                    metrics[f'beta_{period}'] = covariance / market_variance
            
            # Rank correlation (Spearman)
            period = 60
            
            def safe_spearman(x):
                try:
                    if len(x.dropna()) < 2:
                        return np.nan
                    if len(market_returns[-period:].dropna()) < 2:
                        return np.nan
                    return stats.spearmanr(x, market_returns[-period:])[0]
                except:
                    return np.nan
            
            metrics['rank_correlation_60'] = returns.rolling(window=period).apply(safe_spearman)
        else:
            # Default values if no market data - adapt to available data length
            data_length = len(df)
            if data_length >= 20:
                metrics['correlation_spy_20'] = pd.Series(0, index=df.index)
            if data_length >= 60:
                metrics['correlation_spy_60'] = pd.Series(0, index=df.index)
                metrics['beta_60'] = pd.Series(1, index=df.index)
            if data_length >= 252:
                metrics['correlation_spy_252'] = pd.Series(0, index=df.index)
                metrics['beta_252'] = pd.Series(1, index=df.index)
        
        return metrics
    
    @staticmethod
    def calculate_distribution_metrics(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate distribution metrics."""
        metrics = {}
        
        returns = df['close'].pct_change()
        
        # Skewness and Kurtosis - adapt to available data
        data_length = len(df)
        dist_periods = []
        if data_length >= 60:
            dist_periods.append(60)
        if data_length >= 252:
            dist_periods.append(252)
            
        for period in dist_periods:
            if data_length >= period:
                metrics[f'skewness_{period}'] = returns.rolling(window=period).skew()
                metrics[f'kurtosis_{period}'] = returns.rolling(window=period).kurt()
        
        # Jarque-Bera test for normality
        jb_period = min(data_length - 1, 252) if data_length > 60 else min(data_length - 1, 60)
        if jb_period > 30:  # Need reasonable sample for JB test
            def safe_jarque_bera(x):
                try:
                    if len(x.dropna()) < 8:  # JB test needs minimum samples
                        return np.nan
                    return stats.jarque_bera(x.dropna())[0]
                except:
                    return np.nan
            
            metrics[f'jarque_bera_{jb_period}'] = returns.rolling(window=jb_period).apply(safe_jarque_bera)
        
        return metrics
    
    @staticmethod
    def calculate_performance_metrics(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate performance metrics."""
        metrics = {}
        
        returns = df['close'].pct_change()
        
        # Sharpe Ratio - adapt to available data
        data_length = len(df)
        perf_period = min(data_length - 1, 252) if data_length > 60 else min(data_length - 1, 60)
        
        if perf_period > 20:  # Need reasonable sample
            risk_free_rate = 0.05 / 252  # 5% annual
            excess_returns = returns - risk_free_rate
            metrics[f'sharpe_ratio_{perf_period}'] = (
                excess_returns.rolling(window=perf_period).mean() * 252 / 
                (returns.rolling(window=perf_period).std() * np.sqrt(252))
            )
        
            # Sortino Ratio
            downside_returns = returns.copy()
            downside_returns[downside_returns > 0] = 0
            downside_std = downside_returns.rolling(window=perf_period).std() * np.sqrt(252)
            metrics[f'sortino_ratio_{perf_period}'] = (
                excess_returns.rolling(window=perf_period).mean() * 252 / downside_std
            )
            
            # Calmar Ratio
            annual_return = returns.rolling(window=perf_period).mean() * 252
            max_dd_key = f'max_drawdown_{perf_period}'
            # Use safe min operation
            try:
                if len(returns.dropna()) >= perf_period:
                    max_dd = metrics.get(max_dd_key, returns.rolling(window=perf_period, min_periods=1).min())
                else:
                    max_dd = pd.Series(np.nan, index=df.index)
            except Exception:
                max_dd = pd.Series(np.nan, index=df.index)
            
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                metrics['calmar_ratio'] = annual_return / abs(max_dd)
                metrics['calmar_ratio'] = metrics['calmar_ratio'].replace([np.inf, -np.inf], np.nan)
            
            # Information Ratio (simplified - would need benchmark)
            tracking_error = returns.rolling(window=perf_period).std() * np.sqrt(252)
            metrics['information_ratio'] = excess_returns.rolling(window=perf_period).mean() * 252 / tracking_error
        
        return metrics
    
    @staticmethod
    def calculate_rolling_statistics(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate rolling statistics."""
        metrics = {}
        
        returns = df['close'].pct_change()
        
        # Rolling mean returns
        for period in [5, 20, 60]:
            metrics[f'returns_mean_{period}'] = returns.rolling(window=period).mean()
        
        # Rolling standard deviation
        for period in [5, 20, 60]:
            metrics[f'returns_std_{period}'] = returns.rolling(window=period).std()
        
        # Rolling skewness
        for period in [20, 60]:
            metrics[f'returns_skew_{period}'] = returns.rolling(window=period).skew()
        
        # Rolling kurtosis
        for period in [20, 60]:
            metrics[f'returns_kurtosis_{period}'] = returns.rolling(window=period).kurt()
        
        # Volume statistics
        if 'volume' in df.columns:
            for period in [5, 20, 60]:
                if len(df) >= period:
                    metrics[f'volume_mean_{period}'] = df['volume'].rolling(window=period, min_periods=1).mean()
                    metrics[f'volume_std_{period}'] = df['volume'].rolling(window=period, min_periods=1).std()
                else:
                    metrics[f'volume_mean_{period}'] = pd.Series(np.nan, index=df.index)
                    metrics[f'volume_std_{period}'] = pd.Series(np.nan, index=df.index)
        else:
            for period in [5, 20, 60]:
                metrics[f'volume_mean_{period}'] = pd.Series(np.nan, index=df.index)
                metrics[f'volume_std_{period}'] = pd.Series(np.nan, index=df.index)
        
        # High-low spread
        if 'high' in df.columns and 'low' in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                high_low_spread = (df['high'] - df['low']) / df['close']
                high_low_spread = high_low_spread.replace([np.inf, -np.inf], np.nan)
            if len(df) >= 20:
                metrics['high_low_spread_mean_20'] = high_low_spread.rolling(window=20, min_periods=1).mean()
            else:
                metrics['high_low_spread_mean_20'] = pd.Series(np.nan, index=df.index)
        else:
            metrics['high_low_spread_mean_20'] = pd.Series(np.nan, index=df.index)
        
        # Close-to-close volatility
        if len(df) >= 20:
            metrics['close_to_close_volatility_20'] = returns.rolling(window=20, min_periods=1).std() * np.sqrt(252)
        else:
            metrics['close_to_close_volatility_20'] = pd.Series(np.nan, index=df.index)
        
        return metrics
    
    @classmethod
    def calculate_all(cls, df: pd.DataFrame, market_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.Series]:
        """Calculate all statistical metrics."""
        metrics = {}
        
        # Calculate all metric categories with error handling
        try:
            metrics.update(cls.calculate_returns(df))
        except Exception as e:
            logger.debug(f"Error in calculate_returns: {e}")
            
        try:
            metrics.update(cls.calculate_volatility_metrics(df))
        except Exception as e:
            logger.debug(f"Error in calculate_volatility_metrics: {e}")
            
        try:
            metrics.update(cls.calculate_risk_metrics(df))
        except Exception as e:
            logger.debug(f"Error in calculate_risk_metrics: {e}")
            
        try:
            metrics.update(cls.calculate_correlation_metrics(df, market_df))
        except Exception as e:
            logger.debug(f"Error in calculate_correlation_metrics: {e}")
            
        try:
            metrics.update(cls.calculate_distribution_metrics(df))
        except Exception as e:
            logger.debug(f"Error in calculate_distribution_metrics: {e}")
            
        try:
            metrics.update(cls.calculate_performance_metrics(df))
        except Exception as e:
            logger.debug(f"Error in calculate_performance_metrics: {e}")
            
        try:
            metrics.update(cls.calculate_rolling_statistics(df))
        except Exception as e:
            logger.debug(f"Error in calculate_rolling_statistics: {e}")
        
        return metrics