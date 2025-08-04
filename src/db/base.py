"""Database connection and session management."""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import AsyncGenerator

from config.config import settings

# Base for SQLAlchemy models
Base = declarative_base()

# Async engine for async operations
async_engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=10,
    pool_recycle=3600
)

# Sync engine for sync operations
sync_engine = create_engine(
    settings.sync_database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=5
)

# Session makers
async_session_maker = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

sync_session_maker = sessionmaker(
    sync_engine,
    expire_on_commit=False
)

# Dependency for FastAPI
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()

# For backward compatibility
engine = async_engine