#!/usr/bin/env python3
"""Fix polygon volumes by scaling them down by 1M"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.db.base import get_session
from sqlalchemy import update
from src.db.models import OHLCVModel

async def fix_volumes():
    """Scale down polygon volumes by 1M"""
    async for session in get_session():
        # Update all polygon volumes by dividing by 1M
        stmt = (
            update(OHLCVModel)
            .where(OHLCVModel.source == 'polygon')
            .values(volume=OHLCVModel.volume / 1_000_000)
        )
        
        result = await session.execute(stmt)
        await session.commit()
        
        print(f"Updated {result.rowcount} polygon records to scale volume by 1M")
        break

if __name__ == "__main__":
    asyncio.run(fix_volumes())