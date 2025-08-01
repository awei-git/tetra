"""API endpoints for event data (earnings, economic events, holidays)"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from datetime import datetime, date
from typing import List, Optional

from src.db.base import get_session
from src.db.models import EventDataModel, EarningsEventModel, EconomicEventModel
from src.models.event_data import EventType, EventImpact, EventStatus
from src.utils.logging import logger

router = APIRouter()


@router.get("/", response_model=List[dict])
async def get_events(
    event_type: Optional[EventType] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    impact: Optional[int] = Query(None, ge=1, le=4),
    symbol: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_session)
) -> List[dict]:
    """
    Get events with optional filters
    
    - **event_type**: Filter by event type (earnings, economic_release, market_holiday, etc.)
    - **start_date**: Start date for events
    - **end_date**: End date for events
    - **impact**: Filter by impact level (1=low, 2=medium, 3=high, 4=critical)
    - **symbol**: Filter by symbol
    - **source**: Filter by data source (finnhub, yahoo_finance, polygon)
    - **limit**: Maximum number of records to return
    - **offset**: Number of records to skip
    """
    try:
        # Build query
        query = select(EventDataModel)
        
        # Add filters
        filters = []
        if event_type:
            filters.append(EventDataModel.event_type == event_type.value)
        if start_date:
            filters.append(EventDataModel.event_datetime >= datetime.combine(start_date, datetime.min.time()))
        if end_date:
            filters.append(EventDataModel.event_datetime <= datetime.combine(end_date, datetime.max.time()))
        if impact:
            filters.append(EventDataModel.impact == impact)
        if symbol:
            filters.append(EventDataModel.symbol == symbol.upper())
        if source:
            filters.append(EventDataModel.source == source)
        
        if filters:
            query = query.where(and_(*filters))
        
        # Order by datetime descending
        query = query.order_by(EventDataModel.event_datetime.desc())
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
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
            
            # Add full event data if needed
            if event.event_data:
                event_dict["details"] = event.event_data
            
            event_list.append(event_dict)
        
        return event_list
        
    except Exception as e:
        logger.error(f"Error fetching events: {e}")
        raise HTTPException(status_code=500, detail="Error fetching events")


@router.get("/earnings", response_model=List[dict])
async def get_earnings_events(
    symbol: Optional[str] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    has_estimates: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_session)
) -> List[dict]:
    """
    Get earnings events with optional filters
    
    - **symbol**: Filter by stock symbol
    - **start_date**: Start date for earnings
    - **end_date**: End date for earnings
    - **has_estimates**: Filter to only show events with EPS/revenue estimates
    - **limit**: Maximum number of records to return
    """
    try:
        # Join with EventDataModel to get full event info
        query = select(EarningsEventModel, EventDataModel).join(
            EventDataModel, EarningsEventModel.event_id == EventDataModel.id
        )
        
        # Add filters
        filters = []
        if symbol:
            filters.append(EarningsEventModel.symbol == symbol.upper())
        if start_date:
            filters.append(EarningsEventModel.event_datetime >= datetime.combine(start_date, datetime.min.time()))
        if end_date:
            filters.append(EarningsEventModel.event_datetime <= datetime.combine(end_date, datetime.max.time()))
        if has_estimates:
            filters.append(
                (EarningsEventModel.eps_estimate.isnot(None)) | 
                (EarningsEventModel.revenue_estimate.isnot(None))
            )
        
        if filters:
            query = query.where(and_(*filters))
        
        # Order by datetime
        query = query.order_by(EarningsEventModel.event_datetime.desc()).limit(limit)
        
        # Execute query
        result = await db.execute(query)
        earnings = result.all()
        
        # Convert to response format
        earnings_list = []
        for earning, event in earnings:
            earnings_dict = {
                "id": str(earning.id),
                "event_id": str(earning.event_id),
                "symbol": earning.symbol,
                "event_datetime": earning.event_datetime.isoformat(),
                "event_name": event.event_name,
                "impact": event.impact,
                "source": event.source,
                "fiscal_period": earning.fiscal_period,
                "call_time": earning.call_time,
                "eps_actual": float(earning.eps_actual) if earning.eps_actual else None,
                "eps_estimate": float(earning.eps_estimate) if earning.eps_estimate else None,
                "eps_surprise": float(earning.eps_surprise) if earning.eps_surprise else None,
                "eps_surprise_pct": float(earning.eps_surprise_pct) if earning.eps_surprise_pct else None,
                "revenue_actual": float(earning.revenue_actual) if earning.revenue_actual else None,
                "revenue_estimate": float(earning.revenue_estimate) if earning.revenue_estimate else None,
                "revenue_surprise": float(earning.revenue_surprise) if earning.revenue_surprise else None,
                "revenue_surprise_pct": float(earning.revenue_surprise_pct) if earning.revenue_surprise_pct else None,
                "guidance": earning.guidance,
                "created_at": earning.created_at.isoformat()
            }
            earnings_list.append(earnings_dict)
        
        return earnings_list
        
    except Exception as e:
        logger.error(f"Error fetching earnings: {e}")
        raise HTTPException(status_code=500, detail="Error fetching earnings events")


@router.get("/economic", response_model=List[dict])
async def get_economic_events(
    currency: Optional[str] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    min_impact: Optional[int] = Query(None, ge=1, le=4),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_session)
) -> List[dict]:
    """
    Get economic events with optional filters
    
    - **currency**: Filter by currency (USD, EUR, etc.)
    - **start_date**: Start date for events
    - **end_date**: End date for events
    - **min_impact**: Minimum impact level (1=low, 2=medium, 3=high, 4=critical)
    - **limit**: Maximum number of records to return
    """
    try:
        # Join with EventDataModel
        query = select(EconomicEventModel, EventDataModel).join(
            EventDataModel, EconomicEventModel.event_id == EventDataModel.id
        )
        
        # Add filters
        filters = []
        if currency:
            filters.append(EconomicEventModel.currency == currency.upper())
        if start_date:
            filters.append(EconomicEventModel.event_datetime >= datetime.combine(start_date, datetime.min.time()))
        if end_date:
            filters.append(EconomicEventModel.event_datetime <= datetime.combine(end_date, datetime.max.time()))
        if min_impact:
            filters.append(EventDataModel.impact >= min_impact)
        
        if filters:
            query = query.where(and_(*filters))
        
        # Order by datetime
        query = query.order_by(EconomicEventModel.event_datetime.desc()).limit(limit)
        
        # Execute query
        result = await db.execute(query)
        events = result.all()
        
        # Convert to response format
        event_list = []
        for econ_event, event in events:
            event_dict = {
                "id": str(econ_event.id),
                "event_id": str(econ_event.event_id),
                "event_name": event.event_name,
                "event_datetime": econ_event.event_datetime.isoformat(),
                "currency": econ_event.currency,
                "impact": event.impact,
                "source": event.source,
                "actual": float(econ_event.actual) if econ_event.actual else None,
                "forecast": float(econ_event.forecast) if econ_event.forecast else None,
                "previous": float(econ_event.previous) if econ_event.previous else None,
                "revised": float(econ_event.revised) if econ_event.revised else None,
                "unit": econ_event.unit,
                "frequency": econ_event.frequency,
                "created_at": econ_event.created_at.isoformat()
            }
            event_list.append(event_dict)
        
        return event_list
        
    except Exception as e:
        logger.error(f"Error fetching economic events: {e}")
        raise HTTPException(status_code=500, detail="Error fetching economic events")


@router.get("/holidays", response_model=List[dict])
async def get_market_holidays(
    year: Optional[int] = Query(None),
    country: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_session)
) -> List[dict]:
    """
    Get market holidays
    
    - **year**: Filter by year
    - **country**: Filter by country code
    """
    try:
        # Query for market holidays
        query = select(EventDataModel).where(
            EventDataModel.event_type == EventType.MARKET_HOLIDAY.value
        )
        
        # Add filters
        filters = []
        if year:
            filters.append(
                and_(
                    func.extract('year', EventDataModel.event_datetime) == year
                )
            )
        if country:
            filters.append(EventDataModel.country == country.upper())
        
        if filters:
            query = query.where(and_(*filters))
        
        # Order by date
        query = query.order_by(EventDataModel.event_datetime)
        
        # Execute query
        result = await db.execute(query)
        holidays = result.scalars().all()
        
        # Convert to response format
        holiday_list = []
        for holiday in holidays:
            holiday_dict = {
                "id": str(holiday.id),
                "date": holiday.event_datetime.date().isoformat(),
                "name": holiday.event_name,
                "description": holiday.description,
                "country": holiday.country,
                "source": holiday.source,
                "created_at": holiday.created_at.isoformat()
            }
            holiday_list.append(holiday_dict)
        
        return holiday_list
        
    except Exception as e:
        logger.error(f"Error fetching holidays: {e}")
        raise HTTPException(status_code=500, detail="Error fetching market holidays")


@router.get("/stats", response_model=dict)
async def get_event_stats(
    db: AsyncSession = Depends(get_session)
) -> dict:
    """Get statistics about events in the database"""
    try:
        # Total events by type
        type_stats = await db.execute(
            select(
                EventDataModel.event_type,
                func.count(EventDataModel.id).label('count')
            ).group_by(EventDataModel.event_type)
        )
        
        # Events by source
        source_stats = await db.execute(
            select(
                EventDataModel.source,
                func.count(EventDataModel.id).label('count')
            ).group_by(EventDataModel.source)
        )
        
        # Upcoming earnings count
        upcoming_earnings = await db.execute(
            select(func.count(EarningsEventModel.id)).where(
                EarningsEventModel.event_datetime >= datetime.now()
            )
        )
        
        # Build response
        stats = {
            "total_events": sum(row[1] for row in type_stats.all()),
            "by_type": {row[0]: row[1] for row in type_stats.all()},
            "by_source": {row[0]: row[1] for row in source_stats.all()},
            "upcoming_earnings": upcoming_earnings.scalar(),
            "last_updated": datetime.now().isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail="Error fetching statistics")