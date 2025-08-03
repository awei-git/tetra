from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging

from ..services.database import get_db_session
from ..services.monitor import MonitorService
from ..models.responses import CoverageResponse, SchemaResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/coverage", response_model=CoverageResponse)
async def get_coverage(db=Depends(get_db_session)):
    """Get data coverage statistics for all schemas"""
    try:
        monitor_service = MonitorService(db)
        coverage = await monitor_service.get_data_coverage()
        return coverage
    except Exception as e:
        logger.error(f"Error getting coverage: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get coverage data")


@router.get("/schemas", response_model=SchemaResponse)
async def get_schemas(db=Depends(get_db_session)):
    """Get database schema information"""
    try:
        monitor_service = MonitorService(db)
        schemas = await monitor_service.get_schema_info()
        return schemas
    except Exception as e:
        logger.error(f"Error getting schemas: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get schema info")


@router.get("/stats/{schema_name}")
async def get_schema_stats(
    schema_name: str,
    db=Depends(get_db_session)
):
    """Get detailed statistics for a specific schema"""
    try:
        monitor_service = MonitorService(db)
        stats = await monitor_service.get_schema_statistics(schema_name)
        return stats
    except Exception as e:
        logger.error(f"Error getting stats for {schema_name}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get statistics for schema: {schema_name}"
        )


@router.get("/symbols/{schema_name}")
async def get_symbol_details(
    schema_name: str,
    db=Depends(get_db_session)
):
    """Get detailed coverage information for each symbol in a schema"""
    try:
        monitor_service = MonitorService(db)
        symbols = await monitor_service.get_symbol_details(schema_name)
        return {"schema": schema_name, "symbols": symbols}
    except Exception as e:
        logger.error(f"Error getting symbol details for {schema_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get symbol details for schema: {schema_name}"
        )


@router.get("/daily-update-summary")
async def get_daily_update_summary(db=Depends(get_db_session)):
    """Get summary of the most recent daily update run"""
    try:
        monitor_service = MonitorService(db)
        summary = await monitor_service.get_daily_update_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting daily update summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get daily update summary"
        )


@router.post("/trigger-daily-update")
async def trigger_daily_update(db=Depends(get_db_session)):
    """Manually trigger the daily update job"""
    import subprocess
    import asyncio
    from pathlib import Path
    
    try:
        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent.parent
        script_path = project_root / "scripts" / "daily_update.py"
        
        # Run the daily update script in the background
        process = await asyncio.create_subprocess_exec(
            "python", str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(project_root)
        )
        
        # Don't wait for completion - just start it
        return {
            "status": "started",
            "message": "Daily update job has been triggered. Check the summary in a few moments.",
            "pid": process.pid
        }
        
    except Exception as e:
        logger.error(f"Error triggering daily update: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger daily update: {str(e)}"
        )