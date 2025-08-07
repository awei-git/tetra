"""API endpoints for strategy backtest results and rankings."""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
import asyncpg
import logging

from ..services.database import get_db_session
from ..models.responses import StandardResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/strategies", tags=["strategies"])


@router.get("/latest-results")
async def get_latest_results(
    limit: int = Query(20, ge=1, le=100),
    db: asyncpg.Connection = Depends(get_db_session)
) -> StandardResponse:
    """Get the latest strategy backtest results."""
    try:
        # Get the most recent run date
        latest_run = await db.fetchval("""
            SELECT MAX(run_date) FROM strategies.backtest_results
        """)
        
        if not latest_run:
            return StandardResponse(
                success=True,
                data={"results": [], "run_date": None}
            )
        
        # Get results from the latest run
        query = """
            SELECT 
                br.strategy_name,
                br.total_return,
                br.annualized_return,
                br.sharpe_ratio,
                br.max_drawdown,
                br.volatility,
                br.win_rate,
                br.total_trades,
                br.metadata,
                sr.overall_rank,
                sr.composite_score,
                sr.category
            FROM strategies.backtest_results br
            LEFT JOIN strategies.strategy_rankings sr 
                ON br.strategy_name = sr.strategy_name 
                AND br.run_date = sr.run_date
            WHERE br.run_date = $1
            ORDER BY sr.overall_rank ASC NULLS LAST
            LIMIT $2
        """
        
        results = await db.fetch(query, latest_run, limit)
        
        return StandardResponse(
            success=True,
            data={
                "results": [dict(row) for row in results],
                "run_date": latest_run.isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error fetching latest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rankings")
async def get_strategy_rankings(
    run_date: Optional[date] = None,
    category: Optional[str] = None,
    db: asyncpg.Connection = Depends(get_db_session)
) -> StandardResponse:
    """Get strategy rankings for a specific run date."""
    try:
        # If no run date provided, use the latest
        if not run_date:
            run_date = await db.fetchval("""
                SELECT MAX(run_date)::date FROM strategies.strategy_rankings
            """)
            
            if not run_date:
                return StandardResponse(
                    success=True,
                    data={"rankings": [], "run_date": None}
                )
        
        # Build query
        query = """
            SELECT 
                strategy_name,
                overall_rank,
                rank_by_sharpe,
                rank_by_return,
                rank_by_consistency,
                composite_score,
                category
            FROM strategies.strategy_rankings
            WHERE run_date::date = $1
        """
        
        params = [run_date]
        
        if category:
            query += " AND category = $2"
            params.append(category)
        
        query += " ORDER BY overall_rank ASC"
        
        rankings = await db.fetch(query, *params)
        
        return StandardResponse(
            success=True,
            data={
                "rankings": [dict(row) for row in rankings],
                "run_date": run_date.isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error fetching rankings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance-history/{strategy_name}")
async def get_strategy_performance_history(
    strategy_name: str,
    days: int = Query(30, ge=1, le=365),
    db: asyncpg.Connection = Depends(get_db_session)
) -> StandardResponse:
    """Get performance history for a specific strategy."""
    try:
        since_date = datetime.now() - timedelta(days=days)
        
        query = """
            SELECT 
                run_date,
                total_return,
                sharpe_ratio,
                max_drawdown,
                win_rate,
                total_trades
            FROM strategies.backtest_results
            WHERE strategy_name = $1
            AND run_date >= $2
            ORDER BY run_date ASC
        """
        
        history = await db.fetch(query, strategy_name, since_date)
        
        if not history:
            raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found")
        
        return StandardResponse(
            success=True,
            data={
                "strategy_name": strategy_name,
                "history": [dict(row) for row in history]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching performance history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_backtest_summary(
    run_date: Optional[date] = None,
    db: asyncpg.Connection = Depends(get_db_session)
) -> StandardResponse:
    """Get backtest summary statistics."""
    try:
        # If no run date provided, use the latest
        if not run_date:
            run_date = await db.fetchval("""
                SELECT MAX(run_date)::date FROM strategies.backtest_summary
            """)
            
            if not run_date:
                return StandardResponse(
                    success=True,
                    data={"summary": None, "run_date": None}
                )
        
        query = """
            SELECT 
                total_strategies,
                successful_strategies,
                avg_return,
                avg_sharpe,
                avg_max_drawdown,
                best_return,
                worst_return,
                best_sharpe,
                execution_time,
                metadata
            FROM strategies.backtest_summary
            WHERE run_date::date = $1
            ORDER BY run_date DESC
            LIMIT 1
        """
        
        summary = await db.fetchrow(query, run_date)
        
        if not summary:
            return StandardResponse(
                success=True,
                data={"summary": None, "run_date": run_date.isoformat()}
            )
        
        return StandardResponse(
            success=True,
            data={
                "summary": dict(summary),
                "run_date": run_date.isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error fetching backtest summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/equity-curve/{strategy_name}")
async def get_equity_curve(
    strategy_name: str,
    run_date: Optional[date] = None,
    db: asyncpg.Connection = Depends(get_db_session)
) -> StandardResponse:
    """Get equity curve data for a strategy."""
    try:
        # Build query to get the most recent equity curve
        if run_date:
            query = """
                SELECT ec.dates, ec.values
                FROM strategies.equity_curves ec
                JOIN strategies.backtest_results br ON ec.backtest_id = br.id
                WHERE ec.strategy_name = $1
                AND br.run_date::date = $2
                ORDER BY ec.created_at DESC
                LIMIT 1
            """
            params = [strategy_name, run_date]
        else:
            query = """
                SELECT ec.dates, ec.values
                FROM strategies.equity_curves ec
                WHERE ec.strategy_name = $1
                ORDER BY ec.created_at DESC
                LIMIT 1
            """
            params = [strategy_name]
        
        result = await db.fetchrow(query, *params)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Equity curve not found for strategy '{strategy_name}'")
        
        return StandardResponse(
            success=True,
            data={
                "strategy_name": strategy_name,
                "dates": result["dates"],
                "values": result["values"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching equity curve: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_strategy_categories(
    db: asyncpg.Connection = Depends(get_db_session)
) -> StandardResponse:
    """Get all available strategy categories."""
    try:
        query = """
            SELECT DISTINCT category, COUNT(*) as count
            FROM strategies.strategy_metadata
            WHERE category IS NOT NULL
            GROUP BY category
            ORDER BY category
        """
        
        categories = await db.fetch(query)
        
        return StandardResponse(
            success=True,
            data={
                "categories": [
                    {"name": row["category"], "count": row["count"]} 
                    for row in categories
                ]
            }
        )
        
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison")
async def compare_strategies(
    strategies: List[str] = Query(..., description="List of strategy names to compare"),
    days: int = Query(30, ge=1, le=365),
    db: asyncpg.Connection = Depends(get_db_session)
) -> StandardResponse:
    """Compare performance of multiple strategies."""
    try:
        if len(strategies) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 strategies can be compared at once")
        
        since_date = datetime.now() - timedelta(days=days)
        
        query = """
            SELECT 
                strategy_name,
                run_date,
                total_return,
                sharpe_ratio,
                max_drawdown
            FROM strategies.backtest_results
            WHERE strategy_name = ANY($1::text[])
            AND run_date >= $2
            ORDER BY strategy_name, run_date ASC
        """
        
        results = await db.fetch(query, strategies, since_date)
        
        # Group by strategy
        comparison_data = {}
        for row in results:
            strategy = row["strategy_name"]
            if strategy not in comparison_data:
                comparison_data[strategy] = []
            comparison_data[strategy].append({
                "date": row["run_date"].isoformat(),
                "return": row["total_return"],
                "sharpe": row["sharpe_ratio"],
                "drawdown": row["max_drawdown"]
            })
        
        return StandardResponse(
            success=True,
            data={
                "comparison": comparison_data,
                "period_days": days
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))