"""Benchmark pipeline for running strategy backtests end-of-day."""

from typing import Dict, Any, List
from datetime import datetime, timedelta

from src.pipelines.base import Pipeline, PipelineContext
from src.pipelines.benchmark_pipeline.steps import (
    StrategyCollectionStep,
    SimulationSetupStep,
    StrategyBacktestStep,
    MetricsCalculationStep,
    RankingStep,
    ResultStorageStep
)
from src.utils.logging import logger


class BenchmarkPipeline(Pipeline):
    """Pipeline for running benchmark strategy backtests."""
    
    def __init__(self):
        super().__init__(name="BenchmarkPipeline")
        
        # Define pipeline steps
        self.steps = [
            StrategyCollectionStep(),
            SimulationSetupStep(),
            StrategyBacktestStep(),
            MetricsCalculationStep(),
            RankingStep(),
            ResultStorageStep()
        ]
    
    async def setup(self) -> None:
        """Setup pipeline resources."""
        pass
    
    async def run(self, mode: str = "daily", **kwargs) -> Dict[str, Any]:
        """Run the benchmark pipeline.
        
        Args:
            mode: "daily" for regular EOD run, "backfill" for historical
            **kwargs: Additional parameters like date range, strategies to run
            
        Returns:
            Pipeline execution results
        """
        logger.info(f"Starting benchmark pipeline in {mode} mode")
        
        # Set up context
        context = PipelineContext()
        context.data["mode"] = mode
        context.data["run_date"] = kwargs.get("run_date", datetime.now().date())
        
        if mode == "daily":
            # For daily runs, backtest over recent period (e.g., last 3 months)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=90)
            context.data["backtest_start"] = start_date
            context.data["backtest_end"] = end_date
        elif mode == "backfill":
            # For backfill, use provided dates or defaults
            context.data["backtest_start"] = kwargs.get("start_date", 
                                                       datetime.now().date() - timedelta(days=365))
            context.data["backtest_end"] = kwargs.get("end_date", datetime.now().date())
        
        # Strategy filters
        context.data["strategy_filters"] = kwargs.get("strategies", None)  # None means all
        context.data["universe_filter"] = kwargs.get("universe", "core")  # core, all, specific list
        context.data["parallel"] = kwargs.get("parallel", 4)  # Number of parallel backtests
        
        # Run pipeline
        try:
            for step in self.steps:
                logger.info(f"Executing step: {step.name}")
                result = await step.execute(context)
                
                # Store step results in context for next steps
                context.data[f"{step.name}_result"] = result
                
                # Check if step failed
                if result.get("status") == "failed":
                    logger.error(f"Step {step.name} failed: {result.get('error')}")
                    return {
                        "status": "failed",
                        "failed_step": step.name,
                        "error": result.get("error"),
                        "partial_results": context.data
                    }
            
            # Compile final results
            final_results = {
                "status": "success",
                "run_date": context.data["run_date"],
                "mode": mode,
                "strategies_tested": context.data.get("strategies_tested", 0),
                "total_backtests": context.data.get("total_backtests", 0),
                "best_strategy": context.data.get("best_strategy"),
                "metrics_summary": context.data.get("metrics_summary", {}),
                "execution_time": context.data.get("execution_time")
            }
            
            logger.info(f"Benchmark pipeline completed successfully: {final_results}")
            return final_results
            
        except Exception as e:
            logger.error(f"Benchmark pipeline failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "context": context.data
            }