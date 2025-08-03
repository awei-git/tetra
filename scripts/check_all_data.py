#!/usr/bin/env python3
"""Check all volume data by source"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.db.base import get_session
from sqlalchemy import select, func
from src.db.models import OHLCVModel

async def check_data():
    """Check data by source and date range"""
    async for session in get_session():
        # Check data sources and date ranges
        query = select(
            OHLCVModel.source,
            func.count(OHLCVModel.id).label('count'),
            func.min(OHLCVModel.timestamp).label('min_date'),
            func.max(OHLCVModel.timestamp).label('max_date'),
            func.min(OHLCVModel.volume).label('min_volume'),
            func.max(OHLCVModel.volume).label('max_volume')
        ).group_by(OHLCVModel.source)
        
        result = await session.execute(query)
        rows = result.fetchall()
        
        print("\nData Summary by Source:")
        print("-" * 80)
        print(f"{'Source':<10} {'Count':<10} {'Date Range':<30} {'Volume Range':<30}")
        print("-" * 80)
        
        for row in rows:
            date_range = f"{row.min_date.strftime('%Y-%m-%d')} to {row.max_date.strftime('%Y-%m-%d')}"
            vol_range = f"{row.min_volume:,} to {row.max_volume:,}"
            print(f"{row.source:<10} {row.count:<10} {date_range:<30} {vol_range:<30}")
        
        # Check specific problematic dates for META
        print("\n\nMETA Volume Details around 7/22-7/23:")
        print("-" * 60)
        
        query2 = select(
            OHLCVModel.timestamp,
            OHLCVModel.volume,
            OHLCVModel.source,
            OHLCVModel.timeframe
        ).where(
            OHLCVModel.symbol == 'META',
            OHLCVModel.timestamp >= datetime(2025, 7, 21),
            OHLCVModel.timestamp <= datetime(2025, 7, 24)
        ).order_by(OHLCVModel.timestamp)
        
        result2 = await session.execute(query2)
        rows2 = result2.fetchall()
        
        print(f"{'Date':<20} {'Volume':<15} {'Source':<10} {'Timeframe':<10}")
        print("-" * 60)
        
        for row in rows2:
            date_str = row.timestamp.strftime('%Y-%m-%d %H:%M')
            print(f"{date_str:<20} {row.volume:<15,} {row.source:<10} {row.timeframe:<10}")
        
        break

if __name__ == "__main__":
    asyncio.run(check_data())