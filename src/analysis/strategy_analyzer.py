"""Comprehensive strategy performance analyzer with rolling windows and market scenarios."""

import asyncio
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import numpy as np
from enum import Enum
import logging

from src.simulators.historical import HistoricalSimulator
from src.simulators.portfolio import Portfolio
from src.simulators.base import SimulationConfig
from src.strats.benchmark import get_all_benchmarks
from src.utils.logging import logger


class WindowSize(Enum):
    """Standard backtesting window sizes."""
    TWO_WEEKS = 14
    THREE_MONTHS = 90
    ONE_YEAR = 365


class MarketScenario(Enum):
    """Market scenario types for stress testing."""
    NORMAL = "normal"
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    HIGH_VOLATILITY = "high_volatility"
    CRASH = "crash"
    RECOVERY = "recovery"
    SIDEWAYS = "sideways"
    

@dataclass
class PerformanceMetrics:
    """Performance metrics for a single backtest."""
    strategy: str
    symbol: str
    start_date: date
    end_date: date
    window_size: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    recovery_time: int  # Days to recover from max drawdown
    calmar_ratio: float  # Annual return / Max DD
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    success: bool
    error: Optional[str] = None


@dataclass 
class ScenarioResult:
    """Results under a specific market scenario."""
    scenario: MarketScenario
    metrics: PerformanceMetrics
    scenario_params: Dict[str, Any]


class MarketScenarioSimulator:
    """Simulate different market conditions."""
    
    @staticmethod
    def apply_scenario(historical_data: pd.DataFrame, scenario: MarketScenario) -> pd.DataFrame:
        """Apply market scenario transformations to historical data."""
        df = historical_data.copy()
        
        if scenario == MarketScenario.NORMAL:
            return df
            
        elif scenario == MarketScenario.BULL_MARKET:
            # Add 20% annual drift with lower volatility
            daily_drift = 0.20 / 252
            df['close'] = df['close'] * (1 + daily_drift).cumprod()
            df['high'] = df['high'] * (1 + daily_drift).cumprod()
            df['low'] = df['low'] * (1 + daily_drift).cumprod()
            df['open'] = df['open'] * (1 + daily_drift).cumprod()
            
        elif scenario == MarketScenario.BEAR_MARKET:
            # Subtract 20% annual drift with higher volatility
            daily_drift = -0.20 / 252
            volatility_mult = 1.5
            df['close'] = df['close'] * (1 + daily_drift).cumprod()
            df['high'] = df['high'] * (1 + daily_drift * 0.8).cumprod()
            df['low'] = df['low'] * (1 + daily_drift * 1.2).cumprod()
            df['open'] = df['open'] * (1 + daily_drift).cumprod()
            # Increase daily ranges
            df['high'] = df['high'] * volatility_mult
            df['low'] = df['low'] / volatility_mult
            
        elif scenario == MarketScenario.HIGH_VOLATILITY:
            # Increase volatility by 3x without changing drift
            returns = df['close'].pct_change()
            vol_mult = 3.0
            scaled_returns = returns * vol_mult
            df['close'] = df['close'].iloc[0] * (1 + scaled_returns).cumprod()
            # Widen high/low ranges
            df['high'] = df['close'] * (1 + abs(scaled_returns) * 2)
            df['low'] = df['close'] * (1 - abs(scaled_returns) * 2)
            
        elif scenario == MarketScenario.CRASH:
            # Sudden 30% drop over 10 days, then slow recovery
            crash_start = len(df) // 3
            crash_end = crash_start + 10
            recovery_end = crash_start + 60
            
            # Apply crash
            crash_factor = 0.7
            df.loc[df.index[crash_start:crash_end], 'close'] *= np.linspace(1, crash_factor, crash_end - crash_start)
            
            # Slow recovery
            if recovery_end < len(df):
                recovery_factor = 0.85  # Recover to 85% of pre-crash
                df.loc[df.index[crash_end:recovery_end], 'close'] *= np.linspace(
                    crash_factor, recovery_factor, recovery_end - crash_end
                )
                
        elif scenario == MarketScenario.RECOVERY:
            # V-shaped recovery: 20% drop then 40% rally
            mid_point = len(df) // 2
            df.loc[df.index[:mid_point], 'close'] *= np.linspace(1, 0.8, mid_point)
            df.loc[df.index[mid_point:], 'close'] *= np.linspace(0.8, 1.12, len(df) - mid_point)
            
        elif scenario == MarketScenario.SIDEWAYS:
            # Add mean-reverting noise without trend
            returns = np.random.normal(0, 0.01, len(df))  # 1% daily vol, 0 drift
            df['close'] = df['close'].iloc[0] * (1 + returns).cumprod()
            
        # Ensure OHLC relationships are valid
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        return df


class StrategyAnalyzer:
    """Comprehensive strategy analyzer with rolling windows."""
    
    def __init__(self, symbols: List[str], strategies: Optional[Dict[str, Any]] = None):
        self.symbols = symbols
        self.strategies = strategies or get_all_benchmarks()
        self.results: List[PerformanceMetrics] = []
        self.scenario_results: List[ScenarioResult] = []
        self.simulator = HistoricalSimulator()
        
    async def analyze_strategy_full_history(
        self,
        strategy_name: str,
        symbol: str,
        window_size: WindowSize,
        start_year: int = 2015,
        end_year: int = 2025,
        step_days: int = 30  # Roll window every 30 days
    ) -> pd.DataFrame:
        """Analyze strategy performance across full history with rolling windows."""
        
        results = []
        current_date = date(start_year, 1, 1)
        end_date = date(end_year, 12, 31)
        
        while current_date < end_date:
            window_end = current_date + timedelta(days=window_size.value)
            
            # Skip if window extends beyond data
            if window_end > end_date:
                break
                
            try:
                # Run backtest for this window
                metrics = await self._run_single_backtest(
                    strategy_name,
                    symbol,
                    current_date,
                    window_end,
                    window_size.name
                )
                results.append(metrics)
                
            except Exception as e:
                logger.warning(f"Backtest failed for {strategy_name} on {symbol} "
                             f"from {current_date} to {window_end}: {e}")
            
            # Move to next window
            current_date += timedelta(days=step_days)
        
        # Convert to DataFrame for analysis
        if results:
            df = pd.DataFrame([r.__dict__ for r in results])
            df['start_date'] = pd.to_datetime(df['start_date'])
            df.set_index('start_date', inplace=True)
            return df
        
        return pd.DataFrame()
    
    async def _run_single_backtest(
        self,
        strategy_name: str,
        symbol: str,
        start_date: date,
        end_date: date,
        window_name: str
    ) -> PerformanceMetrics:
        """Run a single backtest and calculate comprehensive metrics."""
        
        try:
            # Create portfolio and strategy
            portfolio = Portfolio(initial_cash=100000.0)
            strategy = self.strategies[strategy_name]
            strategy.set_symbols([symbol])
            
            # Run simulation
            result = await self.simulator.run_simulation(
                portfolio=portfolio,
                start_date=start_date,
                end_date=end_date,
                strategy=strategy
            )
            
            # Calculate additional metrics
            returns = result.returns
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
            
            # Calmar Ratio
            calmar = result.annual_return / abs(result.max_drawdown) if result.max_drawdown != 0 else 0
            
            # Average trade return
            avg_trade_return = 0
            best_trade = 0
            worst_trade = 0
            if result.trades:
                trade_returns = [t.get('pnl', 0) / t.get('entry_price', 1) / t.get('quantity', 1) 
                               for t in result.trades if t.get('entry_price', 0) > 0]
                if trade_returns:
                    avg_trade_return = np.mean(trade_returns)
                    best_trade = max(trade_returns)
                    worst_trade = min(trade_returns)
            
            # Recovery time (simplified - days from max DD to recovery)
            recovery_time = self._calculate_recovery_time(result.equity_curve, result.max_drawdown)
            
            return PerformanceMetrics(
                strategy=strategy_name,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                window_size=window_name,
                total_return=result.total_return,
                annual_return=result.annual_return,
                sharpe_ratio=result.sharpe_ratio,
                sortino_ratio=result.sortino_ratio,
                max_drawdown=result.max_drawdown,
                volatility=result.volatility,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                total_trades=result.total_trades,
                avg_trade_return=avg_trade_return,
                best_trade=best_trade,
                worst_trade=worst_trade,
                recovery_time=recovery_time,
                calmar_ratio=calmar,
                var_95=var_95,
                cvar_95=cvar_95,
                success=True
            )
            
        except Exception as e:
            return PerformanceMetrics(
                strategy=strategy_name,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                window_size=window_name,
                total_return=0,
                annual_return=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown=0,
                volatility=0,
                win_rate=0,
                profit_factor=0,
                total_trades=0,
                avg_trade_return=0,
                best_trade=0,
                worst_trade=0,
                recovery_time=0,
                calmar_ratio=0,
                var_95=0,
                cvar_95=0,
                success=False,
                error=str(e)
            )
    
    def _calculate_recovery_time(self, equity_curve: pd.Series, max_drawdown: float) -> int:
        """Calculate days to recover from max drawdown."""
        if equity_curve.empty or max_drawdown == 0:
            return 0
            
        # Find drawdown trough
        cummax = equity_curve.expanding().max()
        drawdown = (equity_curve - cummax) / cummax
        
        # Find max drawdown date
        trough_date = drawdown.idxmin()
        trough_idx = equity_curve.index.get_loc(trough_date)
        
        # Find recovery date (when equity exceeds previous peak)
        prev_peak = cummax.iloc[trough_idx]
        recovery_mask = equity_curve.iloc[trough_idx:] >= prev_peak
        
        if recovery_mask.any():
            recovery_date = recovery_mask.idxmax()
            recovery_days = (recovery_date - trough_date).days
            return recovery_days
        
        # Not recovered yet
        return -1
    
    async def test_market_scenarios(
        self,
        strategy_name: str,
        symbol: str,
        base_start: date,
        base_end: date,
        scenarios: List[MarketScenario]
    ) -> List[ScenarioResult]:
        """Test strategy under different market scenarios."""
        
        results = []
        
        for scenario in scenarios:
            try:
                # Get historical data
                hist_data = await self.simulator.market_replay._load_symbol_data(
                    symbol, base_start, base_end
                )
                
                # Apply scenario transformation
                scenario_data = MarketScenarioSimulator.apply_scenario(hist_data, scenario)
                
                # Create custom simulator with scenario data
                scenario_sim = HistoricalSimulator()
                # Override the data loading to use our scenario data
                scenario_sim.market_replay._data_cache[symbol] = scenario_data
                
                # Run backtest
                portfolio = Portfolio(initial_cash=100000.0)
                strategy = self.strategies[strategy_name]
                strategy.set_symbols([symbol])
                
                result = await scenario_sim.run_simulation(
                    portfolio=portfolio,
                    start_date=base_start,
                    end_date=base_end,
                    strategy=strategy
                )
                
                # Create metrics
                metrics = PerformanceMetrics(
                    strategy=strategy_name,
                    symbol=symbol,
                    start_date=base_start,
                    end_date=base_end,
                    window_size=f"scenario_{scenario.value}",
                    total_return=result.total_return,
                    annual_return=result.annual_return,
                    sharpe_ratio=result.sharpe_ratio,
                    sortino_ratio=result.sortino_ratio,
                    max_drawdown=result.max_drawdown,
                    volatility=result.volatility,
                    win_rate=result.win_rate,
                    profit_factor=result.profit_factor,
                    total_trades=result.total_trades,
                    avg_trade_return=0,  # Simplified
                    best_trade=0,
                    worst_trade=0,
                    recovery_time=0,
                    calmar_ratio=result.annual_return / abs(result.max_drawdown) if result.max_drawdown != 0 else 0,
                    var_95=0,
                    cvar_95=0,
                    success=True
                )
                
                scenario_result = ScenarioResult(
                    scenario=scenario,
                    metrics=metrics,
                    scenario_params={"type": scenario.value}
                )
                
                results.append(scenario_result)
                
            except Exception as e:
                logger.error(f"Scenario test failed for {scenario.value}: {e}")
        
        return results
    
    def analyze_performance_series(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time series of performance metrics."""
        
        if df.empty:
            return {}
        
        analysis = {
            "summary_stats": {
                "mean_return": df['total_return'].mean(),
                "std_return": df['total_return'].std(),
                "min_return": df['total_return'].min(),
                "max_return": df['total_return'].max(),
                "skew": df['total_return'].skew(),
                "kurtosis": df['total_return'].kurt(),
                "positive_periods": (df['total_return'] > 0).sum() / len(df),
                "mean_sharpe": df['sharpe_ratio'].mean(),
                "mean_max_dd": df['max_drawdown'].mean(),
                "worst_dd": df['max_drawdown'].min(),
                "consistency": df['total_return'].std() / abs(df['total_return'].mean()) if df['total_return'].mean() != 0 else float('inf')
            },
            "risk_metrics": {
                "downside_deviation": df[df['total_return'] < 0]['total_return'].std(),
                "var_95": np.percentile(df['total_return'], 5),
                "cvar_95": df[df['total_return'] <= np.percentile(df['total_return'], 5)]['total_return'].mean(),
                "max_consecutive_losses": self._max_consecutive_losses(df['total_return']),
            },
            "regime_analysis": self._analyze_regimes(df),
            "stability_score": self._calculate_stability_score(df)
        }
        
        return analysis
    
    def _max_consecutive_losses(self, returns: pd.Series) -> int:
        """Calculate maximum consecutive losing periods."""
        losing = (returns < 0).astype(int)
        losing_streaks = losing.groupby((losing != losing.shift()).cumsum()).sum()
        return losing_streaks.max() if not losing_streaks.empty else 0
    
    def _analyze_regimes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance in different market regimes."""
        
        # Simple regime detection based on rolling volatility
        df['volatility_regime'] = pd.qcut(df['volatility'], q=3, labels=['low', 'medium', 'high'])
        
        regime_stats = {}
        for regime in ['low', 'medium', 'high']:
            regime_data = df[df['volatility_regime'] == regime]
            if not regime_data.empty:
                regime_stats[f'{regime}_vol'] = {
                    'mean_return': regime_data['total_return'].mean(),
                    'win_rate': (regime_data['total_return'] > 0).mean(),
                    'avg_sharpe': regime_data['sharpe_ratio'].mean()
                }
        
        return regime_stats
    
    def _calculate_stability_score(self, df: pd.DataFrame) -> float:
        """Calculate strategy stability score (0-100)."""
        
        if df.empty or len(df) < 10:
            return 0
        
        # Components of stability
        consistency = min(1 - df['total_return'].std() / (abs(df['total_return'].mean()) + 1e-6), 1)
        win_rate = (df['total_return'] > 0).mean()
        sharpe_consistency = min(1 - df['sharpe_ratio'].std() / (abs(df['sharpe_ratio'].mean()) + 1e-6), 1)
        dd_control = min(1 - abs(df['max_drawdown'].mean()), 1)
        
        # Weighted score
        stability = (
            consistency * 0.3 +
            win_rate * 0.3 +
            sharpe_consistency * 0.2 +
            dd_control * 0.2
        ) * 100
        
        return max(0, min(100, stability))
    
    def rank_strategies(self, all_results: pd.DataFrame) -> pd.DataFrame:
        """Rank strategies based on comprehensive metrics."""
        
        if all_results.empty:
            return pd.DataFrame()
        
        # Group by strategy
        strategy_stats = all_results.groupby('strategy').agg({
            'total_return': ['mean', 'std', 'min', 'max'],
            'sharpe_ratio': 'mean',
            'max_drawdown': ['mean', 'min'],
            'win_rate': 'mean',
            'calmar_ratio': 'mean',
            'total_trades': 'mean'
        }).round(4)
        
        # Flatten column names
        strategy_stats.columns = ['_'.join(col).strip() for col in strategy_stats.columns]
        
        # Calculate composite score
        strategy_stats['return_score'] = strategy_stats['total_return_mean'] * 100
        strategy_stats['risk_adjusted_score'] = strategy_stats['sharpe_ratio_mean'] * 50
        strategy_stats['consistency_score'] = (1 - strategy_stats['total_return_std']) * 30
        strategy_stats['drawdown_score'] = (1 - abs(strategy_stats['max_drawdown_mean'])) * 20
        
        strategy_stats['total_score'] = (
            strategy_stats['return_score'] +
            strategy_stats['risk_adjusted_score'] +
            strategy_stats['consistency_score'] +
            strategy_stats['drawdown_score']
        )
        
        # Rank by total score
        strategy_stats['rank'] = strategy_stats['total_score'].rank(ascending=False)
        
        return strategy_stats.sort_values('rank')


async def run_comprehensive_analysis(
    symbols: List[str] = None,
    strategies: List[str] = None,
    start_year: int = 2020,
    end_year: int = 2025
):
    """Run comprehensive strategy analysis."""
    
    # Default symbols and strategies
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
    
    all_strategies = get_all_benchmarks()
    if strategies is None:
        strategies = list(all_strategies.keys())[:5]  # Top 5 for demo
    
    analyzer = StrategyAnalyzer(symbols, all_strategies)
    
    # Collect all results
    all_results = []
    
    logger.info(f"Starting comprehensive analysis for {len(strategies)} strategies on {len(symbols)} symbols")
    
    # 1. Rolling window analysis for each strategy-symbol pair
    for strategy in strategies:
        for symbol in symbols:
            for window in [WindowSize.TWO_WEEKS, WindowSize.THREE_MONTHS, WindowSize.ONE_YEAR]:
                logger.info(f"Analyzing {strategy} on {symbol} with {window.name} windows...")
                
                try:
                    df = await analyzer.analyze_strategy_full_history(
                        strategy, symbol, window, start_year, end_year
                    )
                    
                    if not df.empty:
                        # Add to results
                        for _, row in df.iterrows():
                            all_results.append(row.to_dict())
                        
                        # Analyze the time series
                        analysis = analyzer.analyze_performance_series(df)
                        
                        logger.info(f"  Mean return: {analysis['summary_stats']['mean_return']:.2%}")
                        logger.info(f"  Win rate: {analysis['summary_stats']['positive_periods']:.1%}")
                        logger.info(f"  Stability score: {analysis['stability_score']:.1f}/100")
                        
                except Exception as e:
                    logger.error(f"Failed to analyze {strategy} on {symbol}: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 2. Market scenario testing (sample)
    logger.info("\nTesting market scenarios...")
    scenarios = list(MarketScenario)
    
    scenario_results = []
    for strategy in strategies[:2]:  # Test first 2 strategies
        for symbol in symbols[:2]:  # On first 2 symbols
            test_start = date(2024, 1, 1)
            test_end = date(2024, 12, 31)
            
            results = await analyzer.test_market_scenarios(
                strategy, symbol, test_start, test_end, scenarios
            )
            
            for result in results:
                logger.info(f"{strategy} on {symbol} in {result.scenario.value}: "
                          f"{result.metrics.total_return:.2%} return, "
                          f"{result.metrics.sharpe_ratio:.2f} Sharpe")
                scenario_results.append(result)
    
    # 3. Rank strategies
    logger.info("\nRanking strategies...")
    rankings = analyzer.rank_strategies(results_df)
    
    return {
        'time_series_results': results_df,
        'scenario_results': scenario_results,
        'rankings': rankings,
        'analyzer': analyzer
    }


def generate_analysis_report(analysis_results: Dict) -> str:
    """Generate comprehensive analysis report."""
    
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE STRATEGY ANALYSIS REPORT")
    report.append("=" * 80)
    
    # Strategy Rankings
    report.append("\n1. STRATEGY RANKINGS")
    report.append("-" * 40)
    
    rankings = analysis_results['rankings']
    if not rankings.empty:
        for idx, (strategy, row) in enumerate(rankings.iterrows()):
            report.append(f"\n#{int(row['rank'])}. {strategy}")
            report.append(f"   Score: {row['total_score']:.1f}")
            report.append(f"   Avg Return: {row['total_return_mean']*100:.1f}% "
                         f"(range: {row['total_return_min']*100:.1f}% to {row['total_return_max']*100:.1f}%)")
            report.append(f"   Avg Sharpe: {row['sharpe_ratio_mean']:.2f}")
            report.append(f"   Avg Max DD: {row['max_drawdown_mean']*100:.1f}%")
            report.append(f"   Win Rate: {row['win_rate_mean']*100:.1f}%")
            
            if idx >= 4:  # Show top 5
                break
    
    # Performance by Window Size
    report.append("\n\n2. PERFORMANCE BY WINDOW SIZE")
    report.append("-" * 40)
    
    df = analysis_results['time_series_results']
    if not df.empty:
        window_stats = df.groupby('window_size').agg({
            'total_return': ['mean', 'std'],
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean'
        }).round(4)
        
        for window in window_stats.index:
            stats = window_stats.loc[window]
            report.append(f"\n{window}:")
            report.append(f"  Avg Return: {stats[('total_return', 'mean')]*100:.1f}% "
                         f"(±{stats[('total_return', 'std')]*100:.1f}%)")
            report.append(f"  Avg Sharpe: {stats[('sharpe_ratio', 'mean')]:.2f}")
            report.append(f"  Avg Max DD: {stats[('max_drawdown', 'mean')]*100:.1f}%")
    
    # Market Scenario Performance
    report.append("\n\n3. MARKET SCENARIO STRESS TESTING")
    report.append("-" * 40)
    
    scenario_results = analysis_results['scenario_results']
    if scenario_results:
        scenario_summary = defaultdict(list)
        for result in scenario_results:
            key = f"{result.metrics.strategy} on {result.metrics.symbol}"
            scenario_summary[key].append({
                'scenario': result.scenario.value,
                'return': result.metrics.total_return,
                'sharpe': result.metrics.sharpe_ratio,
                'max_dd': result.metrics.max_drawdown
            })
        
        for strategy_symbol, scenarios in list(scenario_summary.items())[:5]:
            report.append(f"\n{strategy_symbol}:")
            for s in scenarios:
                report.append(f"  {s['scenario']:15}: {s['return']*100:6.1f}% return, "
                             f"{s['sharpe']:5.2f} Sharpe, {s['max_dd']*100:6.1f}% DD")
    
    # Risk Analysis
    report.append("\n\n4. RISK ANALYSIS")
    report.append("-" * 40)
    
    if not df.empty:
        # Find strategies with best risk metrics
        risk_stats = df.groupby('strategy').agg({
            'max_drawdown': ['mean', 'min'],
            'var_95': 'mean',
            'recovery_time': 'mean'
        }).round(4)
        
        report.append("\nLowest Average Drawdown:")
        low_dd = risk_stats.sort_values(('max_drawdown', 'mean'), descending=True).head(3)
        for strategy, row in low_dd.iterrows():
            report.append(f"  {strategy}: {row[('max_drawdown', 'mean')]*100:.1f}% avg, "
                         f"{row[('max_drawdown', 'min')]*100:.1f}% worst")
    
    # Key Insights
    report.append("\n\n5. KEY INSIGHTS")
    report.append("-" * 40)
    
    if not df.empty:
        # Best performing strategy overall
        best_return = df.groupby('strategy')['total_return'].mean().idxmax()
        best_sharpe = df.groupby('strategy')['sharpe_ratio'].mean().idxmax()
        most_consistent = df.groupby('strategy')['total_return'].std().idxmin()
        
        report.append(f"• Best Average Return: {best_return}")
        report.append(f"• Best Risk-Adjusted: {best_sharpe}")
        report.append(f"• Most Consistent: {most_consistent}")
        
        # Market insights
        symbol_perf = df.groupby('symbol')['total_return'].mean().sort_values(ascending=False)
        report.append(f"\n• Best Performing Assets:")
        for symbol, ret in symbol_perf.head(3).items():
            report.append(f"  - {symbol}: {ret*100:.1f}% avg return")
    
    return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    async def main():
        results = await run_comprehensive_analysis(
            symbols=['SPY', 'QQQ', 'IWM'],
            strategies=['buy_and_hold', 'golden_cross', 'turtle_trading'],
            start_year=2023,
            end_year=2025
        )
        
        report = generate_analysis_report(results)
        print(report)
        
        # Save detailed results
        results['time_series_results'].to_csv('/tmp/strategy_analysis_results.csv', index=False)
        print("\nDetailed results saved to /tmp/strategy_analysis_results.csv")
    
    asyncio.run(main())