"""API endpoints for pipeline monitoring and status."""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, Optional
from datetime import datetime
import asyncpg
import logging

from ..services.database import get_db_session
from ..services.pipeline_monitor import PipelineMonitor
from ..models.responses import StandardResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/status")
async def get_pipeline_status(
    pipeline: Optional[str] = Query(None, description="Specific pipeline name"),
    db: asyncpg.Connection = Depends(get_db_session)
) -> StandardResponse:
    """Get current status of pipelines."""
    try:
        monitor = PipelineMonitor(db)
        statuses = await monitor.get_pipeline_status(pipeline)
        
        # Convert dataclass instances to dicts
        status_dict = {}
        for name, status in statuses.items():
            status_dict[name] = {
                'name': status.name,
                'status': status.status,
                'last_run': status.last_run.isoformat() if status.last_run else None,
                'next_run': status.next_run.isoformat() if status.next_run else None,
                'duration_seconds': status.duration_seconds,
                'records_processed': status.records_processed,
                'error_message': status.error_message
            }
        
        return StandardResponse(
            success=True,
            data={'pipelines': status_dict}
        )
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_pipeline_metrics(
    pipeline: Optional[str] = Query(None, description="Specific pipeline name"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: asyncpg.Connection = Depends(get_db_session)
) -> StandardResponse:
    """Get performance metrics for pipelines."""
    try:
        monitor = PipelineMonitor(db)
        metrics = await monitor.get_pipeline_metrics(pipeline, days)
        
        # Convert dataclass instances to dicts
        metrics_dict = {}
        for name, metric in metrics.items():
            metrics_dict[name] = {
                'name': metric.name,
                'avg_duration_seconds': metric.avg_duration_seconds,
                'success_rate': metric.success_rate,
                'total_runs': metric.total_runs,
                'successful_runs': metric.successful_runs,
                'failed_runs': metric.failed_runs,
                'avg_records_processed': metric.avg_records_processed
            }
        
        return StandardResponse(
            success=True,
            data={
                'metrics': metrics_dict,
                'period_days': days
            }
        )
    except Exception as e:
        logger.error(f"Error getting pipeline metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{pipeline_name}")
async def get_pipeline_history(
    pipeline_name: str,
    limit: int = Query(10, ge=1, le=100, description="Number of recent runs to return"),
    db: asyncpg.Connection = Depends(get_db_session)
) -> StandardResponse:
    """Get execution history for a specific pipeline."""
    try:
        monitor = PipelineMonitor(db)
        history = await monitor.get_execution_history(pipeline_name, limit)
        
        return StandardResponse(
            success=True,
            data={
                'pipeline': pipeline_name,
                'history': history
            }
        )
    except Exception as e:
        logger.error(f"Error getting pipeline history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def check_pipeline_health(
    db: asyncpg.Connection = Depends(get_db_session)
) -> StandardResponse:
    """Check overall health of the pipeline system."""
    try:
        monitor = PipelineMonitor(db)
        health = await monitor.check_pipeline_health()
        
        return StandardResponse(
            success=True,
            data=health
        )
    except Exception as e:
        logger.error(f"Error checking pipeline health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest-results")
async def get_latest_assessment_results(
    db: asyncpg.Connection = Depends(get_db_session)
) -> StandardResponse:
    """Get summary of latest assessment pipeline results."""
    try:
        monitor = PipelineMonitor(db)
        results = await monitor.get_latest_assessment_results()
        
        return StandardResponse(
            success=True,
            data=results
        )
    except Exception as e:
        logger.error(f"Error getting latest assessment results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger/{pipeline_name}")
async def trigger_pipeline(
    pipeline_name: str,
    db: asyncpg.Connection = Depends(get_db_session)
) -> StandardResponse:
    """Manually trigger a pipeline execution."""
    try:
        # Validate pipeline name
        valid_pipelines = ['data', 'scenarios', 'metrics', 'assessment']
        if pipeline_name not in valid_pipelines:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid pipeline name. Must be one of: {', '.join(valid_pipelines)}"
            )
        
        # Check if pipeline is already running
        monitor = PipelineMonitor(db)
        statuses = await monitor.get_pipeline_status(pipeline_name)
        
        if pipeline_name in statuses and statuses[pipeline_name].status == 'running':
            raise HTTPException(
                status_code=409,
                detail=f"Pipeline '{pipeline_name}' is already running"
            )
        
        # Record pipeline start
        run_id = await monitor.record_pipeline_start(pipeline_name)
        
        # Import and run the appropriate pipeline
        if pipeline_name == 'data':
            from src.pipelines.data_pipeline import DataPipeline
            pipeline = DataPipeline()
            await pipeline.run()
        elif pipeline_name == 'scenarios':
            from src.pipelines.scenarios_pipeline import ScenariosPipeline
            pipeline = ScenariosPipeline()
            await pipeline.run()
        elif pipeline_name == 'metrics':
            from src.pipelines.metrics_pipeline import MetricsPipeline
            pipeline = MetricsPipeline()
            await pipeline.run()
        elif pipeline_name == 'assessment':
            from src.pipelines.assessment_pipeline import AssessmentPipeline
            pipeline = AssessmentPipeline()
            await pipeline.run()
        
        # Record successful completion
        await monitor.record_pipeline_end(run_id, 'success')
        
        return StandardResponse(
            success=True,
            data={
                'message': f"Pipeline '{pipeline_name}' triggered successfully",
                'run_id': run_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering pipeline: {e}")
        
        # Record failure if we have a run_id
        if 'run_id' in locals():
            await monitor.record_pipeline_end(run_id, 'failed', error_message=str(e))
        
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedules")
async def get_pipeline_schedules(
    db: asyncpg.Connection = Depends(get_db_session)
) -> StandardResponse:
    """Get scheduling information for all pipelines."""
    try:
        schedules = []
        for key, config in PipelineMonitor.PIPELINES.items():
            schedules.append({
                'pipeline': key,
                'name': config['name'],
                'schedule': config['schedule'],
                'description': f"Runs daily at {config['schedule']}"
            })
        
        return StandardResponse(
            success=True,
            data={'schedules': schedules}
        )
    except Exception as e:
        logger.error(f"Error getting pipeline schedules: {e}")
        raise HTTPException(status_code=500, detail=str(e))