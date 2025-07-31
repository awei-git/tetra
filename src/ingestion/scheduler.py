import asyncio
import subprocess
import sys
from datetime import datetime, time
from typing import List, Callable, Optional, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.ingestion.data_ingester import DataIngester
from src.data_definitions.market_universe import MarketUniverse as Universe
from src.utils.logging import logger
from config import settings


class IngestionScheduler:
    """Manages scheduled data ingestion tasks"""
    
    def __init__(self):
        """Initialize the scheduler"""
        self.scheduler = AsyncIOScheduler()
        self.ingester = DataIngester()
        
        # Use centralized universe
        self.universe = Universe
        self.high_priority_symbols = Universe.get_high_priority_symbols()
    
    def setup_schedules(self):
        """Set up all scheduled tasks based on configuration"""
        
        # Daily comprehensive backfill (3 AM)
        self.scheduler.add_job(
            self._daily_backfill,
            CronTrigger(hour=3, minute=0),
            id="daily_backfill",
            name="Daily Comprehensive Backfill",
            misfire_grace_time=7200  # 2 hour grace period
        )
        
        # Intraday updates during market hours (every 15 minutes)
        self.scheduler.add_job(
            self._intraday_update,
            CronTrigger(
                day_of_week="mon-fri",
                hour="9-16",
                minute="*/15"
            ),
            id="intraday_update",
            name="Intraday Market Update",
            misfire_grace_time=300  # 5 minute grace period
        )
        
        # Weekend crypto update (crypto trades 24/7)
        self.scheduler.add_job(
            self._crypto_update,
            CronTrigger(hour="*/4"),  # Every 4 hours
            id="crypto_update",
            name="Crypto Update",
            misfire_grace_time=1800  # 30 minute grace period
        )
        
        logger.info("Scheduled tasks configured")
    
    async def _daily_backfill(self):
        """Run comprehensive daily backfill using the smart backfill script"""
        logger.info("Starting daily comprehensive backfill")
        
        try:
            # Run the daily backfill script
            result = subprocess.run(
                [sys.executable, "scripts/daily_backfill.py", "--scheduled"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Daily backfill completed successfully")
            else:
                logger.error(f"Daily backfill failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error in daily backfill: {e}")
    
    async def _intraday_update(self):
        """Update intraday data during market hours"""
        logger.info("Starting intraday update")
        
        try:
            # Use high priority symbols from universe
            stats = await self.ingester.update_latest_data(
                symbols=self.high_priority_symbols,
                timeframe="5m"
            )
            logger.info(f"Intraday update completed: {stats}")
            
        except Exception as e:
            logger.error(f"Error in intraday update: {e}")
    
    async def _crypto_update(self):
        """Update crypto data"""
        logger.info("Starting crypto update")
        
        try:
            stats = await self.ingester.update_latest_data(
                symbols=Universe.CRYPTO_SYMBOLS,
                timeframe="1h"
            )
            logger.info(f"Crypto update completed: {stats}")
            
        except Exception as e:
            logger.error(f"Error in crypto update: {e}")
    
    def add_custom_job(
        self,
        func: Callable,
        trigger: Any,
        job_id: str,
        name: str,
        **kwargs
    ):
        """Add a custom scheduled job"""
        self.scheduler.add_job(
            func,
            trigger,
            id=job_id,
            name=name,
            **kwargs
        )
        logger.info(f"Added custom job: {name}")
    
    def start(self):
        """Start the scheduler"""
        self.setup_schedules()
        self.scheduler.start()
        logger.info("Ingestion scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        logger.info("Ingestion scheduler stopped")
    
    def get_jobs(self) -> List[dict]:
        """Get list of scheduled jobs"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
        return jobs


async def main():
    """Run the scheduler standalone"""
    scheduler = IngestionScheduler()
    scheduler.start()
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(60)
            
            # Print job status
            jobs = scheduler.get_jobs()
            logger.info(f"Active jobs: {len(jobs)}")
            for job in jobs:
                logger.info(f"  - {job['name']}: Next run at {job['next_run']}")
                
    except KeyboardInterrupt:
        logger.info("Shutting down scheduler")
        scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())