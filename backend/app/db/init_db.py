#!/usr/bin/env python3
"""Initialize database and create TimescaleDB hypertables"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.db.base import engine, Base
from src.db.models import *
from config import settings


async def create_hypertables():
    """Create TimescaleDB hypertables for time-series data"""
    async with engine.begin() as conn:
        # Create hypertables for time-series data
        hypertables = [
            ("market_data.ohlcv", "timestamp"),
            ("market_data.ticks", "timestamp"),
            ("market_data.quotes", "timestamp"),
            ("events.events", "timestamp"),
            ("derived.technical_indicators", "timestamp"),
            ("derived.signals", "timestamp"),
        ]
        
        for table, time_column in hypertables:
            try:
                await conn.execute(
                    text(f"SELECT create_hypertable('{table}', '{time_column}', if_not_exists => TRUE);")
                )
                print(f"Created hypertable for {table}")
                
                # Set chunk time interval
                await conn.execute(
                    text(f"""
                        SELECT set_chunk_time_interval('{table}', INTERVAL '{settings.timescale_chunk_interval}');
                    """)
                )
                
            except Exception as e:
                print(f"Error creating hypertable for {table}: {e}")


async def create_compression_policies():
    """Set up automatic compression for old data"""
    async with engine.begin() as conn:
        compression_tables = [
            "market_data.ohlcv",
            "market_data.ticks",
            "market_data.quotes",
        ]
        
        for table in compression_tables:
            try:
                # Enable compression
                await conn.execute(
                    text(f"""
                        ALTER TABLE {table} SET (
                            timescaledb.compress,
                            timescaledb.compress_segmentby = 'symbol'
                        );
                    """)
                )
                
                # Add compression policy
                await conn.execute(
                    text(f"""
                        SELECT add_compression_policy('{table}', 
                            INTERVAL '{settings.timescale_compression_after}',
                            if_not_exists => TRUE
                        );
                    """)
                )
                print(f"Added compression policy for {table}")
                
            except Exception as e:
                print(f"Error setting compression for {table}: {e}")


async def create_retention_policies():
    """Set up data retention policies"""
    async with engine.begin() as conn:
        retention_tables = [
            "market_data.ticks",  # Keep tick data for shorter period
        ]
        
        for table in retention_tables:
            try:
                await conn.execute(
                    text(f"""
                        SELECT add_retention_policy('{table}', 
                            INTERVAL '{settings.timescale_retention_period}',
                            if_not_exists => TRUE
                        );
                    """)
                )
                print(f"Added retention policy for {table}")
                
            except Exception as e:
                print(f"Error setting retention policy for {table}: {e}")


async def main():
    print("Initializing TimescaleDB features...")
    
    # Create hypertables
    await create_hypertables()
    
    # Set up compression
    await create_compression_policies()
    
    # Set up retention
    await create_retention_policies()
    
    print("Database initialization complete!")


if __name__ == "__main__":
    asyncio.run(main())