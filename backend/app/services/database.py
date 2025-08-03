import asyncpg
from typing import AsyncGenerator
import logging
from contextlib import asynccontextmanager

from ..config import settings

logger = logging.getLogger(__name__)

# Global connection pool
_db_pool = None


async def init_db():
    """Initialize database connection pool"""
    global _db_pool
    try:
        # Parse the database URL for asyncpg
        db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
        _db_pool = await asyncpg.create_pool(
            db_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise


async def close_db():
    """Close database connection pool"""
    global _db_pool
    if _db_pool:
        await _db_pool.close()
        logger.info("Database connection pool closed")


async def get_db_session():
    """Get database connection from pool"""
    if not _db_pool:
        logger.warning("Database pool not initialized, returning None")
        yield None
        return
    
    async with _db_pool.acquire() as connection:
        yield connection