"""Database connection utilities."""

import asyncpg
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import os

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://tetra_user:tetra_password@localhost:5432/tetra"
)


@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Get a database connection."""
    conn = None
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        yield conn
    finally:
        if conn:
            await conn.close()


async def create_pool():
    """Create a connection pool."""
    return await asyncpg.create_pool(
        DATABASE_URL,
        min_size=5,
        max_size=20
    )