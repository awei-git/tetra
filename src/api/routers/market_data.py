from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from datetime import datetime, timedelta
from typing import List, Optional
from decimal import Decimal

from src.db.base import get_session
from src.models.sqlalchemy import OHLCVModel
from src.models.market_data import OHLCVData
from src.utils.logging import logger

router = APIRouter()


@router.get("/ohlcv/{symbol}", response_model=List[OHLCVData])
async def get_ohlcv(
    symbol: str,
    timeframe: str = Query("1d", regex="^(1m|5m|15m|30m|1h|4h|1d|1w|1M)$"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_session)
) -> List[OHLCVData]:
    """
    Get OHLCV data for a symbol
    
    - **symbol**: Stock/ETF/Crypto symbol (e.g., AAPL, SPY, BTC-USD)
    - **timeframe**: Candle timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
    - **start_date**: Start date for data (ISO format)
    - **end_date**: End date for data (ISO format)
    - **limit**: Maximum number of records to return
    """
    try:
        # Build query
        query = select(OHLCVModel).where(
            and_(
                OHLCVModel.symbol == symbol.upper(),
                OHLCVModel.timeframe == timeframe
            )
        )
        
        # Add date filters
        if start_date:
            query = query.where(OHLCVModel.timestamp >= start_date)
        if end_date:
            query = query.where(OHLCVModel.timestamp <= end_date)
        
        # Order by timestamp descending and limit
        query = query.order_by(OHLCVModel.timestamp.desc()).limit(limit)
        
        # Execute query
        result = await db.execute(query)
        ohlcv_records = result.scalars().all()
        
        # Convert to Pydantic models
        return [OHLCVData.model_validate(record) for record in ohlcv_records]
        
    except Exception as e:
        logger.error(f"Error fetching OHLCV data: {e}")
        raise HTTPException(status_code=500, detail="Error fetching market data")


@router.post("/ohlcv", response_model=OHLCVData)
async def create_ohlcv(
    data: OHLCVData,
    db: AsyncSession = Depends(get_session)
) -> OHLCVData:
    """Create a new OHLCV record"""
    try:
        # Create database model
        db_ohlcv = OHLCVModel(
            symbol=data.symbol.upper(),
            timestamp=data.timestamp,
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            volume=data.volume,
            vwap=data.vwap,
            trades_count=data.trades_count,
            timeframe=data.timeframe,
            source=data.source,
        )
        
        db.add(db_ohlcv)
        await db.commit()
        await db.refresh(db_ohlcv)
        
        return OHLCVData.model_validate(db_ohlcv)
        
    except Exception as e:
        logger.error(f"Error creating OHLCV record: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Error creating market data")


@router.get("/symbols", response_model=List[str])
async def get_symbols(
    db: AsyncSession = Depends(get_session)
) -> List[str]:
    """Get list of all available symbols"""
    try:
        query = select(OHLCVModel.symbol).distinct()
        result = await db.execute(query)
        symbols = result.scalars().all()
        
        return sorted(symbols)
        
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        raise HTTPException(status_code=500, detail="Error fetching symbols")


@router.get("/latest/{symbol}", response_model=OHLCVData)
async def get_latest_price(
    symbol: str,
    timeframe: str = Query("1d", regex="^(1m|5m|15m|30m|1h|4h|1d|1w|1M)$"),
    db: AsyncSession = Depends(get_session)
) -> OHLCVData:
    """Get the latest OHLCV data for a symbol"""
    try:
        query = select(OHLCVModel).where(
            and_(
                OHLCVModel.symbol == symbol.upper(),
                OHLCVModel.timeframe == timeframe
            )
        ).order_by(OHLCVModel.timestamp.desc()).limit(1)
        
        result = await db.execute(query)
        ohlcv = result.scalar_one_or_none()
        
        if not ohlcv:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for symbol {symbol} with timeframe {timeframe}"
            )
        
        return OHLCVData.model_validate(ohlcv)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching latest price: {e}")
        raise HTTPException(status_code=500, detail="Error fetching latest price")