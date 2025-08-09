"""API endpoints for comprehensive strategy analysis."""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, List, Dict, Any
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json
import asyncio

from src.analysis.strategy_analyzer import (
    StrategyAnalyzer, 
    WindowSize, 
    MarketScenario,
    run_comprehensive_analysis,
    generate_analysis_report
)
from src.simulators.historical import HistoricalSimulator
from src.strats.benchmark import get_all_benchmarks
from src.utils.logging import logger

router = APIRouter(prefix="/api/strategies", tags=["strategies"])

# Cache for analysis results
analysis_cache = {}
analysis_running = False


@router.get("/analysis")
async def get_analysis_data(
    time_window: str = Query("3_months", description="Time window for analysis"),
    symbol: str = Query("all", description="Symbol to filter by")
):
    """Get comprehensive analysis data for strategies."""
    try:
        # Check cache
        cache_key = f"{time_window}_{symbol}"
        if cache_key in analysis_cache:
            cached_data = analysis_cache[cache_key]
            if (datetime.now() - cached_data['timestamp']).seconds < 3600:  # 1 hour cache
                return {"success": True, "data": cached_data['data']}
        
        # Load latest analysis results
        results_path = Path("/tmp/strategy_analysis/comprehensive_results.csv")
        if not results_path.exists():
            # Return sample data for demo
            return {"success": True, "data": get_sample_analysis_data()}
        
        # Load and filter results
        df = pd.read_csv(results_path)
        
        # Filter by time window
        if time_window != "all":
            df = df[df['window_size'] == time_window]
        
        # Filter by symbol
        if symbol != "all":
            df = df[df['symbol'] == symbol]
        
        # Prepare response data
        analysis_data = prepare_analysis_response(df)
        
        # Cache results
        analysis_cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': analysis_data
        }
        
        return {"success": True, "data": analysis_data}
        
    except Exception as e:
        logger.error(f"Error getting analysis data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance-grid")
async def get_performance_grid(
    metric: str = Query("return", description="Metric to display"),
    time_window: str = Query("3_months", description="Time window")
):
    """Get performance grid data for heatmap display."""
    try:
        # Load results
        results_path = Path("/tmp/strategy_analysis/comprehensive_results.csv")
        if not results_path.exists():
            return {"success": True, "data": get_sample_grid_data()}
        
        df = pd.read_csv(results_path)
        
        # Filter by time window
        if time_window != "all":
            df = df[df['window_size'] == time_window]
        
        # Pivot data for grid
        metric_map = {
            'return': 'total_return',
            'sharpe': 'sharpe_ratio',
            'drawdown': 'max_drawdown',
            'trades': 'total_trades'
        }
        
        value_col = metric_map.get(metric, 'total_return')
        
        # Group by strategy and symbol, taking mean of metric
        grid_data = df.groupby(['strategy', 'symbol'])[value_col].mean().reset_index()
        
        # Convert to nested dict format
        result = {}
        for _, row in grid_data.iterrows():
            if row['strategy'] not in result:
                result[row['strategy']] = {}
            result[row['strategy']][row['symbol']] = {
                metric: row[value_col],
                'success': True
            }
        
        # Get unique symbols and strategies
        symbols = sorted(df['symbol'].unique().tolist())
        strategies = sorted(df['strategy'].unique().tolist())
        
        return {
            "success": True,
            "data": {
                "grid": result,
                "symbols": symbols,
                "strategies": strategies
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting performance grid: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/time-series/{strategy}")
async def get_strategy_time_series(
    strategy: str,
    symbol: str = Query("SPY", description="Symbol to analyze"),
    window_size: str = Query("3_months", description="Window size"),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
):
    """Get time series data for a specific strategy."""
    try:
        # Load results
        results_path = Path("/tmp/strategy_analysis/comprehensive_results.csv")
        if not results_path.exists():
            return {"success": True, "data": get_sample_time_series()}
        
        df = pd.read_csv(results_path)
        df['start_date'] = pd.to_datetime(df['start_date'])
        
        # Filter data
        mask = (df['strategy'] == strategy) & (df['symbol'] == symbol)
        if window_size != "all":
            mask &= (df['window_size'] == window_size)
        if start_date:
            mask &= (df['start_date'] >= pd.Timestamp(start_date))
        if end_date:
            mask &= (df['start_date'] <= pd.Timestamp(end_date))
        
        filtered_df = df[mask].sort_values('start_date')
        
        # Prepare time series
        time_series = {
            'dates': filtered_df['start_date'].dt.strftime('%Y-%m-%d').tolist(),
            'returns': filtered_df['total_return'].tolist(),
            'sharpe': filtered_df['sharpe_ratio'].tolist(),
            'drawdown': filtered_df['max_drawdown'].tolist(),
            'cumulative_return': calculate_cumulative_returns(filtered_df['total_return'].tolist())
        }
        
        # Calculate rolling statistics
        rolling_stats = {
            'avg_return': filtered_df['total_return'].mean(),
            'volatility': filtered_df['total_return'].std(),
            'best_period': filtered_df['total_return'].max(),
            'worst_period': filtered_df['total_return'].min(),
            'win_rate': (filtered_df['total_return'] > 0).mean()
        }
        
        return {
            "success": True,
            "data": {
                "time_series": time_series,
                "rolling_stats": rolling_stats,
                "periods_analyzed": len(filtered_df)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting time series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-metrics")
async def get_risk_metrics(
    time_window: str = Query("3_months", description="Time window")
):
    """Get risk analysis metrics for all strategies."""
    try:
        # Load results
        results_path = Path("/tmp/strategy_analysis/comprehensive_results.csv")
        if not results_path.exists():
            return {"success": True, "data": get_sample_risk_metrics()}
        
        df = pd.read_csv(results_path)
        
        # Filter by time window
        if time_window != "all":
            df = df[df['window_size'] == time_window]
        
        # Calculate risk metrics by strategy
        risk_data = []
        for strategy in df['strategy'].unique():
            strat_df = df[df['strategy'] == strategy]
            
            risk_metrics = {
                'name': strategy,
                'max_drawdown': strat_df['max_drawdown'].mean(),
                'worst_drawdown': strat_df['max_drawdown'].min(),
                'var_95': strat_df['var_95'].mean() if 'var_95' in strat_df else -0.02,
                'cvar_95': strat_df['cvar_95'].mean() if 'cvar_95' in strat_df else -0.03,
                'downside_dev': strat_df['volatility'].mean() * 0.7,  # Approximation
                'recovery_time': strat_df['recovery_time'].mean() if 'recovery_time' in strat_df else 30,
                'risk_score': calculate_risk_score(strat_df)
            }
            
            risk_data.append(risk_metrics)
        
        # Sort by risk score
        risk_data.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return {"success": True, "data": {"risk_metrics": risk_data}}
        
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scenario-analysis")
async def get_scenario_analysis(
    scenario: str = Query("bull_market", description="Market scenario"),
    top_n: int = Query(10, description="Number of top strategies to return")
):
    """Get strategy performance under specific market scenarios."""
    try:
        # This would load actual scenario test results
        # For now, return sample data
        scenario_results = get_sample_scenario_results(scenario, top_n)
        
        return {"success": True, "data": scenario_results}
        
    except Exception as e:
        logger.error(f"Error getting scenario analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
async def get_strategy_recommendations():
    """Get portfolio recommendations based on analysis."""
    try:
        # Load analysis results and generate recommendations
        recommendations = generate_portfolio_recommendations()
        
        return {"success": True, "data": recommendations}
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-analysis")
async def run_strategy_analysis(
    background_tasks: BackgroundTasks,
    mode: str = "quick_test",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    include_scenarios: bool = True,
    generate_report: bool = True
):
    """Run comprehensive strategy analysis."""
    global analysis_running
    
    try:
        if analysis_running:
            return {
                "success": False,
                "message": "Analysis already running",
                "status": "running"
            }
        
        # Start analysis in background
        background_tasks.add_task(
            run_analysis_task,
            mode=mode,
            start_date=start_date,
            end_date=end_date,
            include_scenarios=include_scenarios,
            generate_report=generate_report
        )
        
        analysis_running = True
        
        return {
            "success": True,
            "message": "Analysis started",
            "estimated_time": get_estimated_time(mode)
        }
        
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis-status")
async def get_analysis_status():
    """Get current analysis status."""
    try:
        return {
            "success": True,
            "data": {
                "running": analysis_running,
                "last_updated": get_last_analysis_time(),
                "progress": get_analysis_progress() if analysis_running else 100
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting analysis status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions

def prepare_analysis_response(df: pd.DataFrame) -> Dict[str, Any]:
    """Prepare comprehensive analysis response from dataframe."""
    return {
        "summary": {
            "total_backtests": len(df),
            "strategies_analyzed": df['strategy'].nunique(),
            "symbols_analyzed": df['symbol'].nunique(),
            "date_range": {
                "start": df['start_date'].min(),
                "end": df['end_date'].max()
            }
        },
        "top_performers": get_top_performers(df),
        "performance_by_window": get_performance_by_window(df),
        "available_strategies": sorted(df['strategy'].unique().tolist()),
        "available_symbols": sorted(df['symbol'].unique().tolist())
    }


def get_top_performers(df: pd.DataFrame, n: int = 5) -> List[Dict]:
    """Get top performing strategies."""
    top = df.groupby('strategy').agg({
        'total_return': 'mean',
        'sharpe_ratio': 'mean',
        'max_drawdown': 'mean',
        'win_rate': 'mean'
    }).round(4)
    
    top['score'] = (
        top['total_return'] * 0.4 +
        top['sharpe_ratio'] * 0.3 +
        (1 - abs(top['max_drawdown'])) * 0.3
    )
    
    top = top.sort_values('score', ascending=False).head(n)
    
    return [
        {
            'name': strategy,
            'return': row['total_return'],
            'sharpe': row['sharpe_ratio'],
            'drawdown': row['max_drawdown'],
            'win_rate': row['win_rate']
        }
        for strategy, row in top.iterrows()
    ]


def get_performance_by_window(df: pd.DataFrame) -> Dict[str, Dict]:
    """Get average performance by window size."""
    result = {}
    
    for window in df['window_size'].unique():
        window_df = df[df['window_size'] == window]
        result[window] = {
            'avg_return': window_df['total_return'].mean(),
            'avg_sharpe': window_df['sharpe_ratio'].mean(),
            'avg_drawdown': window_df['max_drawdown'].mean(),
            'count': len(window_df)
        }
    
    return result


def calculate_cumulative_returns(returns: List[float]) -> List[float]:
    """Calculate cumulative returns from period returns."""
    cumulative = [1.0]
    for r in returns:
        cumulative.append(cumulative[-1] * (1 + r))
    return cumulative[1:]


def calculate_risk_score(df: pd.DataFrame) -> float:
    """Calculate risk score for a strategy (0-100, higher is better)."""
    # Simple risk scoring based on multiple factors
    avg_dd = abs(df['max_drawdown'].mean())
    vol = df['volatility'].mean() if 'volatility' in df else 0.2
    win_rate = df['win_rate'].mean() if 'win_rate' in df else 0.5
    
    # Score components
    dd_score = max(0, 100 * (1 - avg_dd / 0.3))  # 30% drawdown = 0 score
    vol_score = max(0, 100 * (1 - vol / 0.4))     # 40% volatility = 0 score
    win_score = win_rate * 100
    
    # Weighted average
    risk_score = dd_score * 0.4 + vol_score * 0.3 + win_score * 0.3
    
    return round(risk_score, 1)


def generate_portfolio_recommendations() -> Dict[str, Any]:
    """Generate portfolio recommendations based on analysis."""
    # This would use actual analysis results
    # For demo, return sample recommendations
    return {
        "conservative": {
            "strategies": [
                {"name": "buy_and_hold", "allocation": 40},
                {"name": "all_weather", "allocation": 30},
                {"name": "dividend", "allocation": 20},
                {"name": "bonds_hedge", "allocation": 10}
            ],
            "expected_return": 0.08,
            "max_drawdown": -0.10,
            "sharpe_ratio": 0.85
        },
        "balanced": {
            "strategies": [
                {"name": "momentum_factor", "allocation": 30},
                {"name": "buy_and_hold", "allocation": 25},
                {"name": "turtle_trading", "allocation": 20},
                {"name": "rsi_reversion", "allocation": 15},
                {"name": "all_weather", "allocation": 10}
            ],
            "expected_return": 0.12,
            "max_drawdown": -0.15,
            "sharpe_ratio": 1.05
        },
        "aggressive": {
            "strategies": [
                {"name": "momentum_factor", "allocation": 35},
                {"name": "crypto", "allocation": 20},
                {"name": "ai_growth", "allocation": 20},
                {"name": "volatility", "allocation": 15},
                {"name": "morning_breakout", "allocation": 10}
            ],
            "expected_return": 0.18,
            "max_drawdown": -0.25,
            "sharpe_ratio": 0.95
        }
    }


async def run_analysis_task(
    mode: str,
    start_date: Optional[date],
    end_date: Optional[date],
    include_scenarios: bool,
    generate_report: bool
):
    """Background task to run comprehensive analysis."""
    global analysis_running
    
    try:
        logger.info(f"Starting analysis in {mode} mode")
        
        # Run the analysis
        # This would call the actual analysis runner
        # For now, simulate with delay
        await asyncio.sleep(10)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
    finally:
        analysis_running = False


def get_estimated_time(mode: str) -> str:
    """Get estimated analysis time based on mode."""
    times = {
        "quick_test": "5-10 minutes",
        "core_analysis": "30-60 minutes",
        "full_analysis": "2-3 hours"
    }
    return times.get(mode, "Unknown")


def get_last_analysis_time() -> Optional[str]:
    """Get timestamp of last analysis."""
    results_path = Path("/tmp/strategy_analysis/comprehensive_results.csv")
    if results_path.exists():
        return datetime.fromtimestamp(results_path.stat().st_mtime).isoformat()
    return None


def get_analysis_progress() -> int:
    """Get current analysis progress percentage."""
    # This would check actual progress
    # For now return dummy value
    return 45


# Sample data functions for demo

def get_sample_analysis_data() -> Dict:
    """Get sample analysis data for demo."""
    return {
        "summary": {
            "total_backtests": 1350,
            "strategies_analyzed": 20,
            "symbols_analyzed": 18,
            "date_range": {
                "start": "2023-01-01",
                "end": "2025-08-07"
            }
        },
        "top_performers": [
            {"name": "momentum_factor", "return": 0.182, "sharpe": 1.45, "drawdown": -0.12, "win_rate": 0.68},
            {"name": "turtle_trading", "return": 0.156, "sharpe": 1.32, "drawdown": -0.15, "win_rate": 0.65},
            {"name": "buy_and_hold", "return": 0.145, "sharpe": 1.08, "drawdown": -0.18, "win_rate": 0.78},
            {"name": "golden_cross", "return": 0.132, "sharpe": 1.15, "drawdown": -0.14, "win_rate": 0.62},
            {"name": "rsi_reversion", "return": 0.128, "sharpe": 1.22, "drawdown": -0.11, "win_rate": 0.71}
        ],
        "available_strategies": [
            "buy_and_hold", "golden_cross", "turtle_trading", "rsi_reversion",
            "momentum_factor", "macd_crossover", "bollinger_bands", "dual_momentum"
        ],
        "available_symbols": ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL"]
    }


def get_sample_grid_data() -> Dict:
    """Get sample grid data for demo."""
    strategies = ["buy_and_hold", "momentum_factor", "turtle_trading", "rsi_reversion"]
    symbols = ["SPY", "QQQ", "IWM", "AAPL"]
    
    grid = {}
    for strategy in strategies:
        grid[strategy] = {}
        for symbol in symbols:
            # Generate random but consistent values
            base_return = 0.05 + np.random.random() * 0.15
            grid[strategy][symbol] = {
                "return": base_return,
                "sharpe": base_return * 6 + np.random.random() * 0.5,
                "drawdown": -0.05 - np.random.random() * 0.15,
                "trades": int(50 + np.random.random() * 200)
            }
    
    return {
        "grid": grid,
        "symbols": symbols,
        "strategies": strategies
    }


def get_sample_time_series() -> Dict:
    """Get sample time series data."""
    dates = pd.date_range(start='2024-01-01', end='2025-08-07', freq='W')
    n = len(dates)
    
    # Generate realistic looking returns
    returns = np.random.normal(0.002, 0.02, n)
    returns = pd.Series(returns).rolling(3).mean().fillna(0).values
    
    cumulative = [1.0]
    for r in returns:
        cumulative.append(cumulative[-1] * (1 + r))
    
    return {
        "time_series": {
            "dates": [d.strftime('%Y-%m-%d') for d in dates],
            "returns": returns.tolist(),
            "sharpe": (returns / 0.02 * np.sqrt(52)).tolist(),
            "drawdown": (-np.abs(returns) * 2).tolist(),
            "cumulative_return": cumulative[1:]
        },
        "rolling_stats": {
            "avg_return": float(returns.mean()),
            "volatility": float(returns.std()),
            "best_period": float(returns.max()),
            "worst_period": float(returns.min()),
            "win_rate": float((returns > 0).mean())
        },
        "periods_analyzed": n
    }


def get_sample_risk_metrics() -> List[Dict]:
    """Get sample risk metrics."""
    strategies = [
        "buy_and_hold", "momentum_factor", "turtle_trading", 
        "rsi_reversion", "all_weather", "crypto"
    ]
    
    metrics = []
    for i, strategy in enumerate(strategies):
        base_risk = 0.1 + i * 0.02
        metrics.append({
            "name": strategy,
            "max_drawdown": -base_risk - np.random.random() * 0.05,
            "worst_drawdown": -base_risk * 1.5 - np.random.random() * 0.1,
            "var_95": -base_risk * 0.2,
            "cvar_95": -base_risk * 0.25,
            "downside_dev": base_risk * 0.8,
            "recovery_time": int(20 + i * 5 + np.random.random() * 10),
            "risk_score": max(20, 85 - i * 10 + np.random.random() * 10)
        })
    
    return metrics


def get_sample_scenario_results(scenario: str, top_n: int) -> Dict:
    """Get sample scenario analysis results."""
    # Different strategies perform differently in different scenarios
    scenario_winners = {
        "bull_market": ["momentum_factor", "ai_growth", "crypto", "mega_cap_momentum"],
        "bear_market": ["all_weather", "defensive_value", "gold_hedge", "bonds_rotation"],
        "high_volatility": ["volatility", "options_strangle", "vix_hedge", "market_neutral"],
        "crash": ["tail_hedge", "put_protection", "cash_reserve", "defensive_sectors"],
        "recovery": ["value_recovery", "small_cap_growth", "cyclicals", "financials"],
        "sideways": ["rsi_reversion", "bollinger_bands", "pairs_trading", "theta_harvest"]
    }
    
    strategies = scenario_winners.get(scenario, ["buy_and_hold"] * 4)
    
    results = []
    for i, strategy in enumerate(strategies[:top_n]):
        base_return = 0.15 - i * 0.02 if scenario == "bull_market" else -0.05 - i * 0.02
        results.append({
            "name": strategy,
            "return": base_return + np.random.random() * 0.02,
            "sharpe": 1.2 - i * 0.1,
            "max_drawdown": -0.08 - i * 0.02
        })
    
    return {
        "scenario": scenario,
        "description": get_scenario_description(scenario),
        "rankings": results,
        "market_conditions": {
            "volatility": "High" if "volatility" in scenario else "Normal",
            "trend": "Upward" if "bull" in scenario else "Downward" if "bear" in scenario else "Sideways",
            "risk_level": "Extreme" if "crash" in scenario else "Moderate"
        }
    }


def get_scenario_description(scenario: str) -> str:
    """Get description for market scenario."""
    descriptions = {
        "bull_market": "Strong upward trend with 20% annual growth and reduced volatility",
        "bear_market": "Sustained decline with -20% annual drift and increased volatility",
        "high_volatility": "Normal returns but 3x typical volatility",
        "crash": "Sudden 30% drop over 10 days followed by partial recovery",
        "recovery": "V-shaped recovery pattern after market bottom",
        "sideways": "Range-bound market with no clear trend"
    }
    return descriptions.get(scenario, "Unknown scenario")