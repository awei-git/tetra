from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from datetime import datetime
from typing import List, Optional

from src.db.base import get_session
from src.models.sqlalchemy import EventDataModel
from src.models import EventType, EventImpact, EventStatus
from src.utils.logging import logger

router = APIRouter()


@router.get("/", response_model=List[dict])
async def get_events(
    event_type: Optional[EventType] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    impact_level: Optional[str] = Query(None, regex="^(low|medium|high|critical)$"),
    symbol: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    processed: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_session)
) -> List[dict]:
    """
    Get events with optional filters
    
    - **event_type**: Filter by event type
    - **start_date**: Start date for events
    - **end_date**: End date for events
    - **impact_level**: Filter by impact level (low, medium, high, critical)
    - **symbol**: Filter by affected symbol
    - **limit**: Maximum number of records to return
    - **processed**: Filter by processed status
    """
    try:
        # Build query
        query = select(EventDataModel)
        
        # Add filters
        filters = []
        if event_type:
            filters.append(EventDataModel.event_type == event_type.value)
        if start_date:
            filters.append(EventDataModel.event_datetime >= start_date)
        if end_date:
            filters.append(EventDataModel.event_datetime <= end_date)
        if impact_level:
            filters.append(EventDataModel.impact == EventImpact[impact_level.upper()].value)
        if processed is not None:
            # EventDataModel doesn't have processed field
        
        if filters:
            query = query.where(and_(*filters))
        
        # Filter by symbol if provided
        if symbol:
            query = query.where(
                EventDataModel.symbol == symbol.upper()
            )
        
        # Order by timestamp descending and limit
        query = query.order_by(EventDataModel.event_datetime.desc()).limit(limit)
        
        # Execute query
        result = await db.execute(query)
        events = result.scalars().all()
        
        # Convert to response format
        event_list = []
        for event in events:
            event_dict = {
                "id": str(event.id),
                "event_type": event.event_type,
                "event_datetime": event.event_datetime.isoformat(),
                "event_name": event.event_name,
                "description": event.description,
                "impact": event.impact,
                "status": event.status,
                "symbol": event.symbol,
                "currency": event.currency,
                "country": event.country,
                "source": event.source,
                "source_id": event.source_id,
                "created_at": event.created_at.isoformat(),
                "updated_at": event.updated_at.isoformat(),
            }
            
            # Add type-specific data from JSON field
            if event.event_data:
                event_dict["event_data"] = event.event_data
            
            event_list.append(event_dict)
        
        return event_list
        
    except Exception as e:
        logger.error(f"Error fetching events: {e}")
        raise HTTPException(status_code=500, detail="Error fetching events")


@router.post("/economic", response_model=dict)
async def create_economic_event(
    event: EconomicEvent,
    db: AsyncSession = Depends(get_session)
) -> dict:
    """Create a new economic event"""
    try:
        # Prepare event data
        event_data = {
            "indicator": event.indicator,
            "country": event.country,
            "actual": event.actual,
            "forecast": event.forecast,
            "previous": event.previous,
        }
        
        # Create database model
        db_event = EventModel(
            event_type=event.event_type,
            timestamp=event.timestamp,
            title=event.title,
            description=event.description,
            impact_level=event.impact_level,
            affected_symbols=event.affected_symbols,
            affected_sectors=event.affected_sectors,
            source=event.source,
            source_url=event.source_url,
            processed=event.processed,
            processing_notes=event.processing_notes,
            event_data=event_data,
        )
        
        db.add(db_event)
        await db.commit()
        await db.refresh(db_event)
        
        return {
            "id": str(db_event.id),
            "event_type": db_event.event_type.value,
            "timestamp": db_event.timestamp.isoformat(),
            "title": db_event.title,
            "impact_level": db_event.impact_level,
            **event_data
        }
        
    except Exception as e:
        logger.error(f"Error creating economic event: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Error creating event")


@router.post("/security", response_model=dict)
async def create_security_event(
    event: SecurityEvent,
    db: AsyncSession = Depends(get_session)
) -> dict:
    """Create a new security event"""
    try:
        # Prepare event data
        event_data = {
            "symbol": event.symbol,
            "earnings_per_share": event.earnings_per_share,
            "earnings_forecast": event.earnings_forecast,
            "revenue": event.revenue,
            "revenue_forecast": event.revenue_forecast,
            "dividend_amount": event.dividend_amount,
            "ex_dividend_date": event.ex_dividend_date.isoformat() if event.ex_dividend_date else None,
            "split_ratio": event.split_ratio,
        }
        
        # Ensure symbol is in affected_symbols
        if event.symbol not in event.affected_symbols:
            event.affected_symbols.append(event.symbol)
        
        # Create database model
        db_event = EventModel(
            event_type=event.event_type,
            timestamp=event.timestamp,
            title=event.title,
            description=event.description,
            impact_level=event.impact_level,
            affected_symbols=event.affected_symbols,
            affected_sectors=event.affected_sectors,
            source=event.source,
            source_url=event.source_url,
            processed=event.processed,
            processing_notes=event.processing_notes,
            event_data=event_data,
        )
        
        db.add(db_event)
        await db.commit()
        await db.refresh(db_event)
        
        return {
            "id": str(db_event.id),
            "event_type": db_event.event_type.value,
            "timestamp": db_event.timestamp.isoformat(),
            "title": db_event.title,
            "impact_level": db_event.impact_level,
            **event_data
        }
        
    except Exception as e:
        logger.error(f"Error creating security event: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Error creating event")


@router.patch("/{event_id}/process")
async def mark_event_processed(
    event_id: str,
    processing_notes: Optional[str] = None,
    db: AsyncSession = Depends(get_session)
) -> dict:
    """Mark an event as processed"""
    try:
        # Get the event
        query = select(EventModel).where(EventModel.id == event_id)
        result = await db.execute(query)
        event = result.scalar_one_or_none()
        
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        # Update event
        event.processed = True
        if processing_notes:
            event.processing_notes = processing_notes
        event.updated_at = datetime.utcnow()
        
        await db.commit()
        
        return {
            "id": str(event.id),
            "processed": event.processed,
            "processing_notes": event.processing_notes,
            "updated_at": event.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating event: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Error updating event")