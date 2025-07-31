from datetime import datetime
from sqlalchemy import (
    Column, String, DateTime, Numeric, Integer, Boolean, 
    Text, JSON, Index, Float, Enum as SQLEnum,
    UniqueConstraint, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from .base import Base
from ..models.events import EventType
from ..models.derived import IndicatorType, SignalType


class OHLCVModel(Base):
    """OHLCV market data table"""
    __tablename__ = "ohlcv"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", "timeframe", name="uq_ohlcv_symbol_time"),
        Index("idx_ohlcv_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_ohlcv_timestamp", "timestamp"),
        {"schema": "market_data"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    
    open = Column(Numeric(20, 8), nullable=False)
    high = Column(Numeric(20, 8), nullable=False)
    low = Column(Numeric(20, 8), nullable=False)
    close = Column(Numeric(20, 8), nullable=False)
    volume = Column(Integer, nullable=False)
    
    vwap = Column(Numeric(20, 8))
    trades_count = Column(Integer)
    
    timeframe = Column(String(10), nullable=False)
    source = Column(String(50), nullable=False)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=datetime.utcnow)


class TickModel(Base):
    """Trade tick data table"""
    __tablename__ = "ticks"
    __table_args__ = (
        Index("idx_tick_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_tick_timestamp", "timestamp"),
        {"schema": "market_data"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    
    price = Column(Numeric(20, 8), nullable=False)
    size = Column(Integer, nullable=False)
    
    conditions = Column(JSON)
    exchange = Column(String(10))
    
    source = Column(String(50), nullable=False)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class QuoteModel(Base):
    """Quote (bid/ask) data table"""
    __tablename__ = "quotes"
    __table_args__ = (
        Index("idx_quote_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_quote_timestamp", "timestamp"),
        {"schema": "market_data"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    
    bid_price = Column(Numeric(20, 8), nullable=False)
    bid_size = Column(Integer, nullable=False)
    ask_price = Column(Numeric(20, 8), nullable=False)
    ask_size = Column(Integer, nullable=False)
    
    bid_exchange = Column(String(10))
    ask_exchange = Column(String(10))
    
    source = Column(String(50), nullable=False)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class EventModel(Base):
    """Events table for all event types"""
    __tablename__ = "events"
    __table_args__ = (
        Index("idx_event_timestamp", "timestamp"),
        Index("idx_event_type", "event_type"),
        Index("idx_event_affected_symbols", "affected_symbols"),
        {"schema": "events"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    event_type = Column(SQLEnum(EventType), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    
    title = Column(String(500), nullable=False)
    description = Column(Text)
    
    impact_level = Column(String(20), nullable=False)
    
    affected_symbols = Column(JSON, default=list)
    affected_sectors = Column(JSON, default=list)
    
    source = Column(String(50), nullable=False)
    source_url = Column(String(1000))
    
    processed = Column(Boolean, default=False)
    processing_notes = Column(Text)
    
    # Type-specific data stored as JSON
    event_data = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=datetime.utcnow)


class TechnicalIndicatorModel(Base):
    """Technical indicators table"""
    __tablename__ = "technical_indicators"
    __table_args__ = (
        Index("idx_indicator_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_indicator_type", "indicator_type"),
        {"schema": "derived"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    
    indicator_type = Column(SQLEnum(IndicatorType), nullable=False)
    timeframe = Column(String(10), nullable=False)
    
    parameters = Column(JSON, nullable=False)
    
    value = Column(Numeric(20, 8))
    values = Column(JSON, default=dict)
    
    calculation_time = Column(Float, nullable=False)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class SignalModel(Base):
    """Trading signals table"""
    __tablename__ = "signals"
    __table_args__ = (
        Index("idx_signal_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_signal_type", "signal_type"),
        Index("idx_signal_strategy", "strategy_name"),
        {"schema": "derived"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    
    signal_type = Column(SQLEnum(SignalType), nullable=False)
    strength = Column(Float, nullable=False)
    
    strategy_name = Column(String(100), nullable=False)
    timeframe = Column(String(10), nullable=False)
    
    current_price = Column(Numeric(20, 8), nullable=False)
    target_price = Column(Numeric(20, 8))
    stop_price = Column(Numeric(20, 8))
    
    indicators_used = Column(JSON, default=list)
    reasoning = Column(Text)
    
    risk_reward_ratio = Column(Float)
    position_size_suggestion = Column(Float)
    
    backtest_id = Column(String(100))
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class EconomicDataModel(Base):
    """Economic indicators time series data"""
    __tablename__ = "economic_data"
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_econ_symbol_date"),
        Index("idx_econ_symbol_date", "symbol", "date"),
        Index("idx_econ_date", "date"),
        {"schema": "economic_data"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Data identification
    symbol = Column(String(50), nullable=False)  # FRED symbol (e.g., DFF, CPIAUCSL)
    date = Column(DateTime(timezone=True), nullable=False)  # Data point date
    
    # The actual value
    value = Column(Numeric(20, 8), nullable=False)
    
    # Optional metadata
    revision_date = Column(DateTime(timezone=True))  # When this data point was revised
    is_preliminary = Column(Boolean, default=False)  # Preliminary vs final data
    
    # Source tracking
    source = Column(String(50), nullable=False, default="FRED")
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=datetime.utcnow)


class EconomicReleaseModel(Base):
    """Economic data release events for event-driven strategies"""
    __tablename__ = "economic_releases"
    __table_args__ = (
        Index("idx_release_datetime", "release_datetime"),
        Index("idx_release_symbol", "symbol"),
        Index("idx_release_impact", "impact_level"),
        {"schema": "economic_data"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Release identification
    symbol = Column(String(50), nullable=False)  # Primary indicator symbol
    release_name = Column(String(200), nullable=False)  # e.g., "Consumer Price Index"
    
    # Timing
    release_datetime = Column(DateTime(timezone=True), nullable=False)  # Scheduled release time
    period = Column(String(50), nullable=False)  # Period covered (e.g., "2024-01", "Q4 2023")
    
    # Values
    actual = Column(Numeric(20, 8))  # Actual released value
    forecast = Column(Numeric(20, 8))  # Consensus forecast
    previous = Column(Numeric(20, 8))  # Previous period value
    revised_previous = Column(Numeric(20, 8))  # Revised previous value if any
    
    # Market impact
    impact_level = Column(String(20), nullable=False)  # low, medium, high
    surprise_magnitude = Column(Float)  # (actual - forecast) / forecast
    
    # Additional data
    forecast_count = Column(Integer)  # Number of forecasters in consensus
    forecast_std_dev = Column(Numeric(20, 8))  # Standard deviation of forecasts
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class EconomicForecastModel(Base):
    """Economic forecasts for future periods"""
    __tablename__ = "economic_forecasts"
    __table_args__ = (
        UniqueConstraint("symbol", "target_date", "source", "forecast_date", 
                        name="uq_forecast_symbol_target_source_date"),
        Index("idx_forecast_symbol_target", "symbol", "target_date"),
        Index("idx_forecast_date", "forecast_date"),
        {"schema": "economic_data"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Forecast identification
    symbol = Column(String(50), nullable=False)  # Indicator being forecasted
    target_date = Column(DateTime(timezone=True), nullable=False)  # Date being forecasted
    forecast_date = Column(DateTime(timezone=True), nullable=False)  # When forecast was made
    
    # Forecast values
    forecast_value = Column(Numeric(20, 8), nullable=False)
    forecast_low = Column(Numeric(20, 8))  # Low estimate
    forecast_high = Column(Numeric(20, 8))  # High estimate
    
    # Source
    source = Column(String(100), nullable=False)  # e.g., "Fed", "Bloomberg Consensus", "Reuters Poll"
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)