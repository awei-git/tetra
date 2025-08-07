"""Result storage step for benchmark pipeline."""

from typing import Dict, Any, List
from datetime import datetime
import json

from src.pipelines.base import PipelineStep, PipelineContext
from src.utils.logging import logger
from src.utils.db_connection import get_db_connection
from sqlalchemy import text


class ResultStorageStep(PipelineStep[Dict[str, Any]]):
    """Store backtest results and rankings in the database."""
    
    def __init__(self):
        super().__init__(
            name="ResultStorage",
            description="Store strategy backtest results and rankings"
        )
    
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Store results in the database."""
        enhanced_results = context.data.get("enhanced_results", {})
        rankings = context.data.get("rankings", [])
        run_date = datetime.now()
        
        if not enhanced_results:
            return {"status": "failed", "error": "No results to store"}
        
        logger.info(f"Storing results for {len(enhanced_results)} strategies")
        
        try:
            # Store backtest results
            backtest_ids = await self._store_backtest_results(
                enhanced_results, 
                context.data, 
                run_date
            )
            
            # Store rankings
            await self._store_rankings(rankings, run_date)
            
            # Store summary statistics
            await self._store_summary_stats(context.data, run_date)
            
            # Update strategy metadata
            await self._update_strategy_metadata(enhanced_results)
            
            result = {
                "status": "success",
                "results_stored": len(backtest_ids),
                "rankings_stored": len(rankings),
                "run_date": run_date.isoformat()
            }
            
            logger.info(f"Result storage complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to store results: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _store_backtest_results(self, results: Dict[str, Any], 
                                    context_data: Dict, run_date: datetime) -> List[int]:
        """Store individual backtest results."""
        backtest_ids = []
        
        async with get_db_connection() as conn:
            for strategy_name, result in results.items():
                if result.get("status") != "success":
                    continue
                
                # Prepare metadata
                metadata = {
                    "backtest_time": result.get("backtest_time"),
                    "volatility": result.get("volatility"),
                    "sortino_ratio": result.get("sortino_ratio"),
                    "calmar_ratio": result.get("calmar_ratio"),
                    "consistency_score": result.get("consistency_score"),
                    "downside_deviation": result.get("downside_deviation"),
                    "var_95": result.get("var_95"),
                    "cvar_95": result.get("cvar_95"),
                    "avg_trade_duration": result.get("avg_trade_duration"),
                    "win_loss_ratio": result.get("win_loss_ratio"),
                    "expectancy": result.get("expectancy"),
                    "positive_months_pct": result.get("positive_months_pct"),
                    "best_trade": result.get("best_trade"),
                    "worst_trade": result.get("worst_trade"),
                    "profit_factor": result.get("profit_factor")
                }
                
                # Insert backtest result
                query = text("""
                    INSERT INTO strategies.backtest_results (
                        strategy_name, run_date, backtest_start_date, backtest_end_date,
                        universe, initial_capital, final_value, total_return,
                        annualized_return, sharpe_ratio, max_drawdown, volatility,
                        win_rate, total_trades, metadata, created_at
                    ) VALUES (
                        :strategy_name, :run_date, :start_date, :end_date,
                        :universe, :initial_capital, :final_value, :total_return,
                        :annualized_return, :sharpe_ratio, :max_drawdown, :volatility,
                        :win_rate, :total_trades, :metadata, :created_at
                    ) RETURNING id
                """)
                
                result_row = await conn.execute(query, {
                    "strategy_name": strategy_name,
                    "run_date": run_date,
                    "start_date": context_data.get("backtest_start"),
                    "end_date": context_data.get("backtest_end"),
                    "universe": context_data.get("universe_filter", "core"),
                    "initial_capital": context_data.get("simulator_config", {}).initial_capital if context_data.get("simulator_config") else 100000,
                    "final_value": result.get("final_value"),
                    "total_return": result.get("total_return"),
                    "annualized_return": result.get("annualized_return"),
                    "sharpe_ratio": result.get("sharpe_ratio"),
                    "max_drawdown": result.get("max_drawdown"),
                    "volatility": result.get("volatility"),
                    "win_rate": result.get("win_rate"),
                    "total_trades": result.get("total_trades"),
                    "metadata": json.dumps(metadata),
                    "created_at": run_date
                })
                
                backtest_id = result_row.fetchone()[0]
                backtest_ids.append(backtest_id)
                
                # Store equity curve separately if needed
                if result.get("equity_curve"):
                    await self._store_equity_curve(
                        backtest_id, strategy_name, result["equity_curve"], conn
                    )
        
        return backtest_ids
    
    async def _store_rankings(self, rankings: List[Dict], run_date: datetime):
        """Store strategy rankings."""
        async with get_db_connection() as conn:
            for ranking in rankings:
                query = text("""
                    INSERT INTO strategies.strategy_rankings (
                        run_date, strategy_name, rank_by_sharpe, rank_by_return,
                        rank_by_consistency, composite_score, category, overall_rank,
                        created_at
                    ) VALUES (
                        :run_date, :strategy_name, :rank_by_sharpe, :rank_by_return,
                        :rank_by_consistency, :composite_score, :category, :overall_rank,
                        :created_at
                    )
                """)
                
                await conn.execute(query, {
                    "run_date": run_date,
                    "strategy_name": ranking["strategy_name"],
                    "rank_by_sharpe": ranking.get("rank_by_sharpe"),
                    "rank_by_return": ranking.get("rank_by_return"),
                    "rank_by_consistency": ranking.get("rank_by_consistency"),
                    "composite_score": ranking.get("composite_score"),
                    "category": ranking.get("category"),
                    "overall_rank": ranking.get("overall_rank"),
                    "created_at": run_date
                })
    
    async def _store_summary_stats(self, context_data: Dict, run_date: datetime):
        """Store aggregate statistics."""
        aggregate_stats = context_data.get("aggregate_stats", {})
        if not aggregate_stats:
            return
        
        async with get_db_connection() as conn:
            query = text("""
                INSERT INTO strategies.backtest_summary (
                    run_date, total_strategies, successful_strategies,
                    avg_return, avg_sharpe, avg_max_drawdown,
                    best_return, worst_return, best_sharpe,
                    execution_time, metadata, created_at
                ) VALUES (
                    :run_date, :total_strategies, :successful_strategies,
                    :avg_return, :avg_sharpe, :avg_max_drawdown,
                    :best_return, :worst_return, :best_sharpe,
                    :execution_time, :metadata, :created_at
                )
            """)
            
            await conn.execute(query, {
                "run_date": run_date,
                "total_strategies": context_data.get("strategies_tested", 0),
                "successful_strategies": len([r for r in context_data.get("enhanced_results", {}).values() if r.get("status") == "success"]),
                "avg_return": aggregate_stats.get("avg_return"),
                "avg_sharpe": aggregate_stats.get("avg_sharpe"),
                "avg_max_drawdown": aggregate_stats.get("avg_max_drawdown"),
                "best_return": aggregate_stats.get("best_return"),
                "worst_return": aggregate_stats.get("worst_return"),
                "best_sharpe": aggregate_stats.get("best_sharpe"),
                "execution_time": context_data.get("execution_time"),
                "metadata": json.dumps({
                    "universe": context_data.get("universe_filter"),
                    "date_range": {
                        "start": str(context_data.get("backtest_start")),
                        "end": str(context_data.get("backtest_end"))
                    },
                    "top_by_category": context_data.get("top_by_category", {}),
                    "market_condition_rankings": context_data.get("market_condition_rankings", {})
                }),
                "created_at": run_date
            })
    
    async def _update_strategy_metadata(self, results: Dict[str, Any]):
        """Update strategy metadata with latest performance."""
        async with get_db_connection() as conn:
            for strategy_name, result in results.items():
                if result.get("status") != "success":
                    continue
                
                # Check if strategy exists in metadata table
                check_query = text("""
                    SELECT id FROM strategies.strategy_metadata 
                    WHERE strategy_name = :strategy_name
                """)
                
                existing = await conn.execute(check_query, {"strategy_name": strategy_name})
                
                if existing.fetchone():
                    # Update existing
                    update_query = text("""
                        UPDATE strategies.strategy_metadata
                        SET last_backtest_date = :run_date,
                            last_sharpe_ratio = :sharpe,
                            last_total_return = :return,
                            updated_at = :updated_at
                        WHERE strategy_name = :strategy_name
                    """)
                    
                    await conn.execute(update_query, {
                        "strategy_name": strategy_name,
                        "run_date": datetime.now(),
                        "sharpe": result.get("sharpe_ratio"),
                        "return": result.get("total_return"),
                        "updated_at": datetime.now()
                    })
                else:
                    # Insert new
                    insert_query = text("""
                        INSERT INTO strategies.strategy_metadata (
                            strategy_name, category, description,
                            last_backtest_date, last_sharpe_ratio,
                            last_total_return, created_at
                        ) VALUES (
                            :strategy_name, :category, :description,
                            :run_date, :sharpe, :return, :created_at
                        )
                    """)
                    
                    await conn.execute(insert_query, {
                        "strategy_name": strategy_name,
                        "category": self._get_strategy_category(strategy_name),
                        "description": f"Benchmark strategy: {strategy_name}",
                        "run_date": datetime.now(),
                        "sharpe": result.get("sharpe_ratio"),
                        "return": result.get("total_return"),
                        "created_at": datetime.now()
                    })
    
    async def _store_equity_curve(self, backtest_id: int, strategy_name: str, 
                                equity_curve: Dict, conn):
        """Store equity curve data for visualization."""
        # Store only daily snapshots to limit data size
        if not equity_curve:
            return
        
        # Sample equity curve if too large
        dates = list(equity_curve.keys())
        values = list(equity_curve.values())
        
        # Store max 252 points (1 year of daily data)
        if len(dates) > 252:
            step = len(dates) // 252
            dates = dates[::step]
            values = values[::step]
        
        query = text("""
            INSERT INTO strategies.equity_curves (
                backtest_id, strategy_name, dates, values, created_at
            ) VALUES (
                :backtest_id, :strategy_name, :dates, :values, :created_at
            )
        """)
        
        await conn.execute(query, {
            "backtest_id": backtest_id,
            "strategy_name": strategy_name,
            "dates": json.dumps(dates),
            "values": json.dumps(values),
            "created_at": datetime.now()
        })
    
    def _get_strategy_category(self, strategy_name: str) -> str:
        """Determine strategy category from name."""
        name_lower = strategy_name.lower()
        
        if "trend" in name_lower or "momentum" in name_lower:
            return "trend_following"
        elif "mean_reversion" in name_lower or "pairs" in name_lower:
            return "mean_reversion"
        elif "volatility" in name_lower:
            return "volatility"
        elif "buy_hold" in name_lower:
            return "passive"
        elif "ml" in name_lower or "ai" in name_lower:
            return "machine_learning"
        else:
            return "other"