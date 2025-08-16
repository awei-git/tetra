"""
Stochastic scenario generation step.
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, date, timedelta
from src.pipelines.base import PipelineStep, PipelineContext

logger = logging.getLogger(__name__)


class StochasticScenarioStep(PipelineStep):
    """Generate stochastic scenarios using Monte Carlo and bootstrap methods."""
    
    def __init__(self):
        super().__init__("Stochastic Scenarios", "Generate stochastic scenarios")
    
    async def execute(self, context: PipelineContext) -> Any:
        """Generate stochastic scenarios."""
        logger.info("Generating stochastic scenarios...")
        
        # Get configuration
        config = context.data.get('stochastic_config', {})
        num_scenarios = config.get('num_scenarios', 100)
        scenario_length = config.get('scenario_length_days', 252)  # 1 year default
        methods = config.get('methods', ['monte_carlo', 'bootstrap'])
        
        # Get market data for calibration
        market_data = context.data.get('market_data', {})
        
        stochastic_scenarios = []
        
        # Generate scenarios using each method
        if 'monte_carlo' in methods:
            mc_scenarios = self._generate_monte_carlo_scenarios(
                market_data=market_data,
                num_scenarios=num_scenarios // len(methods),
                scenario_length=scenario_length
            )
            stochastic_scenarios.extend(mc_scenarios)
            logger.info(f"Generated {len(mc_scenarios)} Monte Carlo scenarios")
        
        if 'bootstrap' in methods:
            bootstrap_scenarios = self._generate_bootstrap_scenarios(
                market_data=market_data,
                num_scenarios=num_scenarios // len(methods),
                scenario_length=scenario_length
            )
            stochastic_scenarios.extend(bootstrap_scenarios)
            logger.info(f"Generated {len(bootstrap_scenarios)} bootstrap scenarios")
        
        if 'gbm' in methods:
            gbm_scenarios = self._generate_gbm_scenarios(
                market_data=market_data,
                num_scenarios=num_scenarios // len(methods),
                scenario_length=scenario_length
            )
            stochastic_scenarios.extend(gbm_scenarios)
            logger.info(f"Generated {len(gbm_scenarios)} GBM scenarios")
        
        # Add regime-based scenarios
        if config.get('include_regime_switching', False):
            regime_scenarios = self._generate_regime_switching_scenarios(
                market_data=market_data,
                num_scenarios=10,  # Fewer regime scenarios
                scenario_length=scenario_length
            )
            stochastic_scenarios.extend(regime_scenarios)
            logger.info(f"Generated {len(regime_scenarios)} regime-switching scenarios")
        
        # Add to context
        existing_scenarios = context.data.get('scenarios', [])
        existing_scenarios.extend(stochastic_scenarios)
        context.data['scenarios'] = existing_scenarios
        
        context.set_metric('stochastic_scenarios_generated', len(stochastic_scenarios))
        logger.info(f"Generated {len(stochastic_scenarios)} total stochastic scenarios")
        
        return context
    
    def _generate_monte_carlo_scenarios(
        self,
        market_data: Dict[str, pd.DataFrame],
        num_scenarios: int,
        scenario_length: int
    ) -> List[Dict[str, Any]]:
        """Generate Monte Carlo scenarios based on historical statistics."""
        
        scenarios = []
        
        # Calculate historical statistics
        stats = self._calculate_market_statistics(market_data)
        
        for i in range(num_scenarios):
            scenario = {
                'id': f'monte_carlo_{i}',
                'name': f'Monte Carlo Scenario {i+1}',
                'type': 'stochastic',
                'scenario_type': 'monte_carlo',
                'start_date': date.today(),
                'end_date': date.today() + timedelta(days=scenario_length),
                'description': 'Monte Carlo simulation based on historical parameters',
                'metadata': {
                    'method': 'monte_carlo',
                    'seed': i,
                    'parameters': stats
                },
                'data': self._generate_mc_paths(stats, scenario_length, seed=i)
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_bootstrap_scenarios(
        self,
        market_data: Dict[str, pd.DataFrame],
        num_scenarios: int,
        scenario_length: int
    ) -> List[Dict[str, Any]]:
        """Generate bootstrap scenarios by resampling historical returns."""
        
        scenarios = []
        
        # Extract historical returns
        historical_returns = self._extract_historical_returns(market_data)
        
        if not historical_returns:
            logger.warning("No historical data available for bootstrap")
            return scenarios
        
        for i in range(num_scenarios):
            scenario = {
                'id': f'bootstrap_{i}',
                'name': f'Bootstrap Scenario {i+1}',
                'type': 'stochastic',
                'scenario_type': 'bootstrap',
                'start_date': date.today(),
                'end_date': date.today() + timedelta(days=scenario_length),
                'description': 'Bootstrap resampling of historical returns',
                'metadata': {
                    'method': 'bootstrap',
                    'seed': i,
                    'block_size': 20  # Block bootstrap with 20-day blocks
                },
                'data': self._generate_bootstrap_paths(
                    historical_returns, scenario_length, seed=i
                )
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_gbm_scenarios(
        self,
        market_data: Dict[str, pd.DataFrame],
        num_scenarios: int,
        scenario_length: int
    ) -> List[Dict[str, Any]]:
        """Generate Geometric Brownian Motion scenarios."""
        
        scenarios = []
        
        # Calculate drift and volatility parameters
        params = self._calculate_gbm_parameters(market_data)
        
        for i in range(num_scenarios):
            scenario = {
                'id': f'gbm_{i}',
                'name': f'GBM Scenario {i+1}',
                'type': 'stochastic',
                'scenario_type': 'gbm',
                'start_date': date.today(),
                'end_date': date.today() + timedelta(days=scenario_length),
                'description': 'Geometric Brownian Motion simulation',
                'metadata': {
                    'method': 'gbm',
                    'seed': i,
                    'parameters': params
                },
                'data': self._generate_gbm_paths(params, scenario_length, seed=i)
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_regime_switching_scenarios(
        self,
        market_data: Dict[str, pd.DataFrame],
        num_scenarios: int,
        scenario_length: int
    ) -> List[Dict[str, Any]]:
        """Generate scenarios with regime switching between bull/bear/neutral."""
        
        scenarios = []
        
        # Define market regimes
        regimes = {
            'bull': {'return': 0.15, 'volatility': 0.12, 'persistence': 0.85},
            'bear': {'return': -0.20, 'volatility': 0.25, 'persistence': 0.80},
            'neutral': {'return': 0.07, 'volatility': 0.15, 'persistence': 0.90}
        }
        
        for i in range(num_scenarios):
            scenario = {
                'id': f'regime_switch_{i}',
                'name': f'Regime Switching Scenario {i+1}',
                'type': 'stochastic',
                'scenario_type': 'regime_switching',
                'start_date': date.today(),
                'end_date': date.today() + timedelta(days=scenario_length),
                'description': 'Markov regime-switching model simulation',
                'metadata': {
                    'method': 'regime_switching',
                    'seed': i,
                    'regimes': regimes
                },
                'data': self._generate_regime_paths(regimes, scenario_length, seed=i)
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _calculate_market_statistics(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate statistical parameters from market data."""
        
        stats = {
            'mean_returns': {},
            'volatilities': {},
            'correlations': {}
        }
        
        # Default values if no data
        if not market_data:
            return {
                'mean_returns': {'SPY': 0.0007},  # ~18% annual
                'volatilities': {'SPY': 0.015},  # ~24% annual
                'correlations': {}
            }
        
        # Calculate for each symbol
        returns_df = pd.DataFrame()
        
        for symbol, df in market_data.items():
            if not df.empty and 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                stats['mean_returns'][symbol] = float(returns.mean())
                stats['volatilities'][symbol] = float(returns.std())
                returns_df[symbol] = returns
        
        # Calculate correlations
        if not returns_df.empty and len(returns_df.columns) > 1:
            corr_matrix = returns_df.corr()
            stats['correlations'] = corr_matrix.to_dict()
        
        return stats
    
    def _extract_historical_returns(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Extract historical returns from market data."""
        
        returns = {}
        
        for symbol, df in market_data.items():
            if not df.empty and 'close' in df.columns:
                symbol_returns = df['close'].pct_change().dropna().values
                if len(symbol_returns) > 0:
                    returns[symbol] = symbol_returns
        
        # Use synthetic returns if no data
        if not returns:
            returns['SPY'] = np.random.normal(0.0007, 0.015, 1000)
        
        return returns
    
    def _calculate_gbm_parameters(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate GBM parameters (drift and volatility)."""
        
        params = {}
        
        # Default parameters
        default_drift = 0.08  # 8% annual
        default_vol = 0.20  # 20% annual
        
        if not market_data:
            return {'SPY': {'drift': default_drift, 'volatility': default_vol}}
        
        for symbol, df in market_data.items():
            if not df.empty and 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                
                # Annualized parameters
                annual_return = returns.mean() * 252
                annual_vol = returns.std() * np.sqrt(252)
                
                params[symbol] = {
                    'drift': float(annual_return),
                    'volatility': float(annual_vol)
                }
        
        return params
    
    def _generate_mc_paths(
        self,
        stats: Dict[str, Any],
        scenario_length: int,
        seed: int = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate Monte Carlo price paths."""
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate dates
        dates = pd.date_range(
            start=date.today(),
            periods=scenario_length,
            freq='B'
        )
        
        paths = {}
        
        for symbol in stats['mean_returns'].keys():
            mean = stats['mean_returns'][symbol]
            vol = stats['volatilities'][symbol]
            
            # Generate returns
            returns = np.random.normal(mean, vol, scenario_length)
            
            # Convert to prices
            prices = 100 * np.cumprod(1 + returns)
            
            # Create path data
            path_data = [
                {
                    'date': d.isoformat(),
                    'close': float(p),
                    'returns': float(r),
                    'volatility': float(vol)
                }
                for d, p, r in zip(dates, prices, returns)
            ]
            
            paths[symbol] = path_data
        
        return paths
    
    def _generate_bootstrap_paths(
        self,
        historical_returns: Dict[str, np.ndarray],
        scenario_length: int,
        seed: int = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate bootstrap resampled paths."""
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate dates
        dates = pd.date_range(
            start=date.today(),
            periods=scenario_length,
            freq='B'
        )
        
        paths = {}
        block_size = 20  # Use block bootstrap
        
        for symbol, hist_returns in historical_returns.items():
            # Block bootstrap
            num_blocks = scenario_length // block_size + 1
            returns = []
            
            for _ in range(num_blocks):
                # Random starting point for block
                start_idx = np.random.randint(0, len(hist_returns) - block_size)
                block = hist_returns[start_idx:start_idx + block_size]
                returns.extend(block)
            
            # Trim to scenario length
            returns = returns[:scenario_length]
            
            # Convert to prices
            prices = 100 * np.cumprod(1 + np.array(returns))
            
            # Calculate rolling volatility
            vol = pd.Series(returns).rolling(20, min_periods=1).std().fillna(np.std(returns)).values
            
            # Create path data
            path_data = [
                {
                    'date': d.isoformat(),
                    'close': float(p),
                    'returns': float(r),
                    'volatility': float(v)
                }
                for d, p, r, v in zip(dates, prices, returns, vol)
            ]
            
            paths[symbol] = path_data
        
        return paths
    
    def _generate_gbm_paths(
        self,
        params: Dict[str, Dict[str, float]],
        scenario_length: int,
        seed: int = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate Geometric Brownian Motion paths."""
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate dates
        dates = pd.date_range(
            start=date.today(),
            periods=scenario_length,
            freq='B'
        )
        
        paths = {}
        dt = 1/252  # Daily time step
        
        for symbol, param in params.items():
            drift = param['drift']
            vol = param['volatility']
            
            # GBM simulation
            S0 = 100  # Initial price
            prices = [S0]
            
            for _ in range(scenario_length - 1):
                dW = np.random.normal(0, np.sqrt(dt))
                S_t = prices[-1]
                dS = S_t * (drift * dt + vol * dW)
                prices.append(S_t + dS)
            
            prices = np.array(prices)
            
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])  # First return is 0
            
            # Create path data
            path_data = [
                {
                    'date': d.isoformat(),
                    'close': float(p),
                    'returns': float(r),
                    'volatility': float(vol / np.sqrt(252))  # Daily volatility
                }
                for d, p, r in zip(dates, prices, returns)
            ]
            
            paths[symbol] = path_data
        
        return paths
    
    def _generate_regime_paths(
        self,
        regimes: Dict[str, Dict[str, float]],
        scenario_length: int,
        seed: int = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate regime-switching paths."""
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate dates
        dates = pd.date_range(
            start=date.today(),
            periods=scenario_length,
            freq='B'
        )
        
        # Transition probabilities
        regime_names = list(regimes.keys())
        current_regime = np.random.choice(regime_names)
        
        returns = []
        regime_history = []
        
        for _ in range(scenario_length):
            # Get current regime parameters
            regime_params = regimes[current_regime]
            
            # Generate return for this day
            daily_return = regime_params['return'] / 252  # Convert annual to daily
            daily_vol = regime_params['volatility'] / np.sqrt(252)
            
            day_return = np.random.normal(daily_return, daily_vol)
            returns.append(day_return)
            regime_history.append(current_regime)
            
            # Possibly switch regime
            if np.random.random() > regime_params['persistence']:
                # Switch to different regime
                other_regimes = [r for r in regime_names if r != current_regime]
                current_regime = np.random.choice(other_regimes)
        
        # Convert to prices
        prices = 100 * np.cumprod(1 + np.array(returns))
        
        # Create path data
        path_data = [
            {
                'date': d.isoformat(),
                'close': float(p),
                'returns': float(r),
                'regime': regime,
                'volatility': float(regimes[regime]['volatility'] / np.sqrt(252))
            }
            for d, p, r, regime in zip(dates, prices, returns, regime_history)
        ]
        
        return {'SPY': path_data}  # Return market proxy