#!/usr/bin/env python3
"""Check volume data in database"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.db.base import get_session
from sqlalchemy import select
from src.db.models import OHLCVModel

async def check_volume():
    """Check META volume data"""
    async for session in get_session():
        # Query META data from 7/22/25 onwards
        query = select(
            OHLCVModel.symbol,
            OHLCVModel.timestamp,
            OHLCVModel.close,
            OHLCVModel.volume,
            OHLCVModel.source
        ).where(
            OHLCVModel.symbol == 'META',
            OHLCVModel.timestamp >= datetime(2025, 7, 22)
        ).order_by(OHLCVModel.timestamp.desc()).limit(15)
        
        result = await session.execute(query)
        rows = result.fetchall()
        
        print(f"{'Date':<12} {'Close':<8} {'Volume':<15} {'Source':<10}")
        print("-" * 50)
        
        for row in rows:
            date_str = row.timestamp.strftime('%Y-%m-%d')
            print(f"{date_str:<12} ${row.close:<7.2f} {row.volume:<15,} {row.source:<10}")
        
        break

if __name__ == "__main__":
    asyncio.run(check_volume())