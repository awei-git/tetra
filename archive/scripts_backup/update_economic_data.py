#!/usr/bin/env python3
"""
Update economic indicators from FRED
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipelines.data_pipeline.steps.economic_data import EconomicDataStep
from src.pipelines.base import PipelineContext
from src.utils.logging import logger


async def main():
    """Update stale economic indicators"""
    logger.info("=== Updating Economic Data ===")
    
    # Create economic data step
    econ_step = EconomicDataStep()
    
    # Run in daily mode to get latest data
    context = PipelineContext(data={
        "mode": "daily",
        "start_date": datetime.now().date() - timedelta(days=365),
        "end_date": datetime.now().date()
    })
    
    try:
        result = await econ_step.execute(context)
        records = result.get("total_records", 0)
        logger.info(f"\nTotal records updated: {records}")
        
        # Show details
        if "indicators_updated" in result:
            logger.info("\nIndicators updated:")
            for indicator, count in result["indicators_updated"].items():
                logger.info(f"  {indicator}: {count} records")
                
    except Exception as e:
        logger.error(f"Economic data update failed: {e}")
    
    # Check status
    logger.info("\nChecking economic data status...")
    import subprocess
    subprocess.run([
        sys.executable, "-c",
        """
import psycopg2
from datetime import datetime

conn = psycopg2.connect(host='localhost', port=5432, database='tetra', user='tetra_user', password='tetra_password')
cur = conn.cursor()

# Key indicators to check
indicators = ['DFF', 'DGS10', 'DEXUSEU', 'CPIAUCSL', 'UNRATE', 'GDPC1', 'UMCSENT']

cur.execute('''
    SELECT symbol, MAX(date)::date as last_date, COUNT(*) as records
    FROM economic_data.economic_data
    WHERE symbol = ANY(%s)
    GROUP BY symbol
    ORDER BY symbol
''', (indicators,))

print('\\nKey Economic Indicators:')
print('-' * 50)
for symbol, last_date, records in cur.fetchall():
    days_ago = (datetime.now().date() - last_date).days
    status = '✓' if days_ago <= 31 else f'⚠️  {days_ago}d old'
    print(f'{symbol:12} {last_date} ({records:4} records) {status}')
"""
    ])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Update failed: {e}")