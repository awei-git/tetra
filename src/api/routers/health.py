from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any
import redis.asyncio as aioredis
from datetime import datetime

from src.db.base import get_session
from config import settings

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "tetra-api",
        "version": "1.0.0",
    }


@router.get("/ready")
async def readiness_check(
    db: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Readiness check - verifies all dependencies are available"""
    checks = {
        "database": False,
        "redis": False,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    # Check database
    try:
        result = await db.execute(text("SELECT 1"))
        checks["database"] = result.scalar() == 1
    except Exception as e:
        checks["database_error"] = str(e)
    
    # Check Redis
    try:
        redis = await aioredis.from_url(settings.redis_url)
        await redis.ping()
        checks["redis"] = True
        await redis.close()
    except Exception as e:
        checks["redis_error"] = str(e)
    
    # Overall status
    checks["ready"] = all([checks["database"], checks["redis"]])
    
    return checks


@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """Liveness check - simple endpoint to verify service is running"""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}