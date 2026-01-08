"""Async database engine and session helpers."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from config.config import settings

DATABASE_URL = settings.database_url

engine: AsyncEngine = create_async_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_connection() -> AsyncConnection:
    async with engine.begin() as conn:
        yield conn


async def get_session() -> AsyncSession:
    async with SessionLocal() as session:
        yield session
