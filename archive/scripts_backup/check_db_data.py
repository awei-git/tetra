"""Quick script to check what data is available in the database."""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.db.sync_base import get_session
from src.db.models import OHLCVModel, EventDataModel
from sqlalchemy import func

def check_database_data():
    """Check what data is available in the database."""
    session = get_session()
    
    try:
        # Check OHLCV data
        print("="*60)
        print("OHLCV Market Data Summary")
        print("="*60)
        
        # Total records
        total_ohlcv = session.query(func.count(OHLCVModel.id)).scalar()
        print(f"Total OHLCV records: {total_ohlcv:,}")
        
        if total_ohlcv > 0:
            # Get unique symbols
            symbols = session.query(OHLCVModel.symbol).distinct().limit(20).all()
            print(f"\nSymbols available (first 20): {[s[0] for s in symbols]}")
            
            # Get date range
            min_date = session.query(func.min(OHLCVModel.timestamp)).scalar()
            max_date = session.query(func.max(OHLCVModel.timestamp)).scalar()
            print(f"\nDate range: {min_date} to {max_date}")
            
            # Get timeframes
            timeframes = session.query(OHLCVModel.timeframe).distinct().all()
            print(f"\nTimeframes available: {[t[0] for t in timeframes]}")
            
            # Sample data for a popular symbol
            sample_symbol = 'AAPL'
            sample_count = session.query(func.count(OHLCVModel.id)).filter(
                OHLCVModel.symbol == sample_symbol
            ).scalar()
            
            if sample_count > 0:
                print(f"\n{sample_symbol} has {sample_count:,} records")
                
                # Get latest 5 records
                latest = session.query(OHLCVModel).filter(
                    OHLCVModel.symbol == sample_symbol
                ).order_by(OHLCVModel.timestamp.desc()).limit(5).all()
                
                print(f"\nLatest 5 {sample_symbol} records:")
                for record in latest:
                    print(f"  {record.timestamp}: O={record.open} H={record.high} L={record.low} C={record.close} V={record.volume}")
        
        # Check Event data
        print("\n" + "="*60)
        print("Event Data Summary")
        print("="*60)
        
        total_events = session.query(func.count(EventDataModel.id)).scalar()
        print(f"Total event records: {total_events:,}")
        
        if total_events > 0:
            # Get event types
            event_types = session.query(EventDataModel.event_type).distinct().all()
            print(f"\nEvent types: {[e[0] for e in event_types]}")
            
            # Get date range
            min_date = session.query(func.min(EventDataModel.event_datetime)).scalar()
            max_date = session.query(func.max(EventDataModel.event_datetime)).scalar()
            print(f"\nDate range: {min_date} to {max_date}")
            
            # Sample recent events
            recent_events = session.query(EventDataModel).order_by(
                EventDataModel.event_datetime.desc()
            ).limit(5).all()
            
            print("\nRecent events:")
            for event in recent_events:
                print(f"  {event.event_datetime} - {event.symbol}: {event.event_name} ({event.event_type})")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"Error checking database: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        session.close()


if __name__ == "__main__":
    check_database_data()