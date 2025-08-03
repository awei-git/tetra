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