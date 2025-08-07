"""Metrics calculation step for benchmark pipeline."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from datetime import datetime

from src.pipelines.base import PipelineStep, PipelineContext
from src.utils.logging import logger


class MetricsCalculationStep(PipelineStep[Dict[str, Any]]):
    """Calculate comprehensive metrics for all strategy backtests."""
    
    def __init__(self):
        super().__init__(
            name="MetricsCalculation",
            description="Calculate additional metrics and risk measures"
        )
    
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Calculate comprehensive metrics for each strategy."""
        backtest_results = context.data.get("backtest_results", {})
        simulator_config = context.data.get("simulator_config")
        
        logger.info(f"Calculating metrics for {len(backtest_results)} strategies")
        
        enhanced_results = {}
        
        for name, results in backtest_results.items():
            if results.get("status") != "success":
                enhanced_results[name] = results
                continue
            
            # Calculate additional metrics
            enhanced = results.copy()
            
            # Risk-adjusted metrics
            enhanced["sortino_ratio"] = self._calculate_sortino_ratio(
                results.get("equity_curve"), 
                simulator_config.initial_capital if simulator_config else 100000
            )
            enhanced["calmar_ratio"] = self._calculate_calmar_ratio(
                results.get("annualized_return", 0),
                results.get("max_drawdown", 0)
            )
            
            # Consistency metrics
            if results.get("equity_curve"):
                enhanced["consistency_score"] = self._calculate_consistency_score(results["equity_curve"])
                enhanced["monthly_returns"] = self._calculate_monthly_returns(results["equity_curve"])
                enhanced["positive_months_pct"] = self._calculate_positive_months_pct(enhanced["monthly_returns"])
            
            # Risk metrics
            enhanced["downside_deviation"] = self._calculate_downside_deviation(results.get("equity_curve"))
            enhanced["var_95"] = self._calculate_var(results.get("equity_curve"), 0.95)
            enhanced["cvar_95"] = self._calculate_cvar(results.get("equity_curve"), 0.95)
            
            # Trading efficiency
            if results.get("total_trades", 0) > 0:
                enhanced["avg_trade_duration"] = self._calculate_avg_trade_duration(results.get("trades", []))
                enhanced["win_loss_ratio"] = results.get("avg_win", 0) / abs(results.get("avg_loss", 1)) if results.get("avg_loss") else 0
                enhanced["expectancy"] = self._calculate_expectancy(results)
            
            enhanced_results[name] = enhanced
        
        # Store enhanced results
        context.data["enhanced_results"] = enhanced_results
        
        # Calculate aggregate statistics
        successful_strategies = [r for r in enhanced_results.values() if r.get("status") == "success"]
        
        if successful_strategies:
            aggregate_stats = {
                "avg_return": np.mean([s["total_return"] for s in successful_strategies]),
                "avg_sharpe": np.mean([s["sharpe_ratio"] for s in successful_strategies if s["sharpe_ratio"] is not None]),
                "avg_max_drawdown": np.mean([s["max_drawdown"] for s in successful_strategies]),
                "best_return": max([s["total_return"] for s in successful_strategies]),
                "worst_return": min([s["total_return"] for s in successful_strategies]),
                "most_trades": max([s["total_trades"] for s in successful_strategies]),
                "best_sharpe": max([s["sharpe_ratio"] for s in successful_strategies if s["sharpe_ratio"] is not None]),
            }
            context.data["aggregate_stats"] = aggregate_stats
        
        result = {
            "status": "success",
            "strategies_processed": len(enhanced_results),
            "metrics_calculated": list(next(iter(enhanced_results.values())).keys()) if enhanced_results else []
        }
        
        logger.info(f"Metrics calculation complete")
        return result
    
    def _calculate_sortino_ratio(self, equity_curve: Dict, initial_capital: float) -> float:
        """Calculate Sortino ratio (uses downside deviation instead of total volatility)."""
        if not equity_curve:
            return None
        
        try:
            values = list(equity_curve.values())
            returns = pd.Series(values).pct_change().dropna()
            
            if len(returns) < 2:
                return None
            
            # Calculate downside returns only
            downside_returns = returns[returns < 0]
            
            if len(downside_returns) == 0:
                return float('inf')  # No downside
            
            downside_std = downside_returns.std()
            
            if downside_std == 0:
                return float('inf')
            
            # Annualized return
            total_return = (values[-1] / initial_capital - 1)
            days = len(values)
            annualized_return = (1 + total_return) ** (252 / days) - 1
            
            # Sortino = (Return - Risk Free Rate) / Downside Deviation
            # Assuming 0% risk-free rate
            sortino = annualized_return / (downside_std * np.sqrt(252))
            
            return round(sortino, 2)
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return None
    
    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        if max_drawdown == 0 or max_drawdown is None:
            return None
        return round(annualized_return / abs(max_drawdown), 2)
    
    def _calculate_consistency_score(self, equity_curve: Dict) -> float:
        """Calculate how consistently the strategy generates returns."""
        if not equity_curve:
            return 0
        
        try:
            values = list(equity_curve.values())
            returns = pd.Series(values).pct_change().dropna()
            
            if len(returns) < 20:
                return 0
            
            # Calculate rolling sharpe (20-day windows)
            rolling_sharpe = returns.rolling(20).apply(
                lambda x: x.mean() / x.std() if x.std() > 0 else 0
            ).dropna()
            
            # Consistency is the percentage of positive rolling sharpe periods
            consistency = (rolling_sharpe > 0).sum() / len(rolling_sharpe)
            
            return round(consistency, 2)
            
        except Exception:
            return 0
    
    def _calculate_monthly_returns(self, equity_curve: Dict) -> List[float]:
        """Calculate monthly returns from equity curve."""
        if not equity_curve:
            return []
        
        try:
            df = pd.DataFrame(list(equity_curve.items()), columns=['date', 'value'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Resample to monthly
            monthly = df.resample('M').last()
            monthly_returns = monthly.pct_change().dropna()
            
            return monthly_returns['value'].tolist()
            
        except Exception:
            return []
    
    def _calculate_positive_months_pct(self, monthly_returns: List[float]) -> float:
        """Calculate percentage of positive months."""
        if not monthly_returns:
            return 0
        positive = sum(1 for r in monthly_returns if r > 0)
        return round(positive / len(monthly_returns), 2)
    
    def _calculate_downside_deviation(self, equity_curve: Dict) -> float:
        """Calculate downside deviation."""
        if not equity_curve:
            return None
        
        try:
            values = list(equity_curve.values())
            returns = pd.Series(values).pct_change().dropna()
            downside_returns = returns[returns < 0]
            
            if len(downside_returns) == 0:
                return 0
            
            return round(downside_returns.std() * np.sqrt(252), 4)
            
        except Exception:
            return None
    
    def _calculate_var(self, equity_curve: Dict, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        if not equity_curve:
            return None
        
        try:
            values = list(equity_curve.values())
            returns = pd.Series(values).pct_change().dropna()
            
            if len(returns) < 20:
                return None
            
            var = np.percentile(returns, (1 - confidence_level) * 100)
            return round(var, 4)
            
        except Exception:
            return None
    
    def _calculate_cvar(self, equity_curve: Dict, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if not equity_curve:
            return None
        
        try:
            values = list(equity_curve.values())
            returns = pd.Series(values).pct_change().dropna()
            
            if len(returns) < 20:
                return None
            
            var = np.percentile(returns, (1 - confidence_level) * 100)
            cvar = returns[returns <= var].mean()
            
            return round(cvar, 4)
            
        except Exception:
            return None
    
    def _calculate_avg_trade_duration(self, trades: List[Dict]) -> float:
        """Calculate average trade duration in days."""
        if not trades:
            return 0
        
        try:
            durations = []
            for trade in trades:
                if 'entry_date' in trade and 'exit_date' in trade:
                    entry = pd.to_datetime(trade['entry_date'])
                    exit = pd.to_datetime(trade['exit_date'])
                    duration = (exit - entry).days
                    durations.append(duration)
            
            return round(np.mean(durations), 1) if durations else 0
            
        except Exception:
            return 0
    
    def _calculate_expectancy(self, results: Dict) -> float:
        """Calculate trade expectancy (average profit per trade)."""
        win_rate = results.get("win_rate", 0)
        avg_win = results.get("avg_win", 0)
        avg_loss = results.get("avg_loss", 0)
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        return round(expectancy, 4)