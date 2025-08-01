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
from ..models.event_data import EventType
from ..models.derived import IndicatorType, SignalType
from ..models.news_sentiment import NewsSource, NewsCategory


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


# Event Data Models
class EventDataModel(Base):
    """Base event data storage"""
    __tablename__ = "event_data"
    __table_args__ = (
        Index("idx_event_datetime", "event_datetime"),
        Index("idx_event_type_datetime", "event_type", "event_datetime"),
        Index("idx_symbol_datetime", "symbol", "event_datetime"),
        Index("idx_currency_datetime", "currency", "event_datetime"),
        UniqueConstraint("source", "source_id", name="uq_source_event"),
        {"schema": "events"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event identification
    event_type = Column(String(50), nullable=False)
    event_datetime = Column(DateTime(timezone=True), nullable=False)
    event_name = Column(String(500), nullable=False)
    description = Column(Text)
    
    # Impact and status
    impact = Column(Integer, nullable=False, default=2)  # 1=Low, 2=Medium, 3=High, 4=Critical
    status = Column(String(20), nullable=False, default="scheduled")
    
    # Related entities
    symbol = Column(String(50))  # For company-specific events
    currency = Column(String(10))  # For economic events
    country = Column(String(10))  # Country code
    
    # Source information
    source = Column(String(100), nullable=False)
    source_id = Column(String(255))  # ID in source system
    
    # JSON field for type-specific data
    event_data = Column(JSON, nullable=False, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class EconomicEventModel(Base):
    """Economic event specific data"""
    __tablename__ = "economic_events"
    __table_args__ = (
        Index("idx_econ_event_id", "event_id"),
        Index("idx_econ_datetime", "event_datetime"),
        {"schema": "events"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(UUID(as_uuid=True), ForeignKey("events.event_data.id"), nullable=False)
    
    # Copy key fields for faster queries
    event_datetime = Column(DateTime(timezone=True), nullable=False)
    currency = Column(String(10))
    
    # Economic data
    actual = Column(Numeric(20, 8))
    forecast = Column(Numeric(20, 8))
    previous = Column(Numeric(20, 8))
    revised = Column(Numeric(20, 8))
    
    # Additional context
    unit = Column(String(50))  # %, billions, etc.
    frequency = Column(String(50))  # monthly, quarterly
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class EarningsEventModel(Base):
    """Earnings event specific data"""
    __tablename__ = "earnings_events"
    __table_args__ = (
        Index("idx_earn_event_id", "event_id"),
        Index("idx_earn_symbol_datetime", "symbol", "event_datetime"),
        {"schema": "events"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(UUID(as_uuid=True), ForeignKey("events.event_data.id"), nullable=False)
    
    # Copy key fields for faster queries
    symbol = Column(String(50), nullable=False)
    event_datetime = Column(DateTime(timezone=True), nullable=False)
    
    # Earnings data
    eps_actual = Column(Numeric(20, 8))
    eps_estimate = Column(Numeric(20, 8))
    eps_surprise = Column(Numeric(20, 8))
    eps_surprise_pct = Column(Numeric(20, 8))
    
    # Revenue data
    revenue_actual = Column(Numeric(20, 8))
    revenue_estimate = Column(Numeric(20, 8))
    revenue_surprise = Column(Numeric(20, 8))
    revenue_surprise_pct = Column(Numeric(20, 8))
    
    # Additional info
    guidance = Column(Text)
    call_time = Column(String(10))  # BMO, AMC
    fiscal_period = Column(String(20))  # Q1 2024, FY 2024
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


# News and Sentiment Models
class NewsArticleModel(Base):
    """News articles storage"""
    __tablename__ = "news_articles"
    __table_args__ = (
        Index("idx_news_published", "published_at"),
        Index("idx_news_symbols", "symbols"),
        Index("idx_news_source", "source"),
        UniqueConstraint("source", "source_id", name="uq_source_article"),
        {"schema": "news"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Article identifiers
    source_id = Column(String(500))  # ID from source API
    source = Column(String(100), nullable=False)
    source_category = Column(String(50), nullable=False)
    
    # Article content
    author = Column(String(200))
    title = Column(Text, nullable=False)
    description = Column(Text)
    content = Column(Text)
    url = Column(Text, nullable=False)
    image_url = Column(Text)
    
    # Timing
    published_at = Column(DateTime(timezone=True), nullable=False)
    fetched_at = Column(DateTime(timezone=True), nullable=False)
    
    # Related entities
    symbols = Column(JSON)  # List of stock symbols
    entities = Column(JSON)  # List of entities mentioned
    categories = Column(JSON)  # List of categories
    
    # Raw data
    raw_data = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    
    # Relationship to sentiment
    sentiments = relationship("NewsSentimentModel", back_populates="article", cascade="all, delete-orphan")


class NewsSentimentModel(Base):
    """News sentiment analysis results"""
    __tablename__ = "news_sentiments"
    __table_args__ = (
        Index("idx_sentiment_article", "article_id"),
        Index("idx_sentiment_analyzed", "analyzed_at"),
        Index("idx_sentiment_symbols", "symbols"),
        {"schema": "news"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Link to article
    article_id = Column(UUID(as_uuid=True), ForeignKey("news.news_articles.id"), nullable=False)
    
    # Sentiment scores
    polarity = Column(Float, nullable=False)  # -1 to 1
    subjectivity = Column(Float, nullable=False)  # 0 to 1
    positive = Column(Float, nullable=False)  # 0 to 1
    negative = Column(Float, nullable=False)  # 0 to 1
    neutral = Column(Float, nullable=False)  # 0 to 1
    bullish = Column(Float)  # 0 to 1 (optional)
    bearish = Column(Float)  # 0 to 1 (optional)
    
    # Analysis metadata
    sentiment_model = Column(String(100), nullable=False)
    analyzed_at = Column(DateTime(timezone=True), nullable=False)
    
    # Trading signals
    symbols = Column(JSON)  # Symbols this sentiment applies to
    relevance_score = Column(Float, default=0.0)  # 0 to 1
    impact_score = Column(Float, default=0.0)  # 0 to 1
    
    # Flags
    is_breaking = Column(Boolean, default=False)
    is_rumor = Column(Boolean, default=False)
    requires_confirmation = Column(Boolean, default=False)
    
    # Additional sentiment data (by type)
    sentiment_by_type = Column(JSON)  # Dict of sentiment scores by type
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    
    # Relationship
    article = relationship("NewsArticleModel", back_populates="sentiments")


class NewsClusterModel(Base):
    """Clusters of related news articles"""
    __tablename__ = "news_clusters"
    __table_args__ = (
        Index("idx_cluster_created", "created_at"),
        Index("idx_cluster_symbols", "symbols"),
        {"schema": "news"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Articles in cluster
    article_ids = Column(JSON, nullable=False)  # List of article IDs
    lead_article_id = Column(UUID(as_uuid=True), nullable=False)
    
    # Cluster metadata
    symbols = Column(JSON)
    categories = Column(JSON)
    
    # Timing
    earliest_published = Column(DateTime(timezone=True), nullable=False)
    latest_published = Column(DateTime(timezone=True), nullable=False)
    
    # Aggregated sentiment
    avg_polarity = Column(Float, nullable=False)
    sentiment_std = Column(Float, nullable=False)
    article_count = Column(Integer, nullable=False)
    
    # Cluster properties
    coherence_score = Column(Float, nullable=False)  # 0 to 1
    velocity = Column(Float, nullable=False)  # Articles per hour
    is_trending = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class NewsSummaryModel(Base):
    """Daily/periodic summaries of news sentiment"""
    __tablename__ = "news_summaries"
    __table_args__ = (
        Index("idx_summary_symbol_date", "symbol", "start_date"),
        UniqueConstraint("symbol", "start_date", "end_date", name="uq_summary_period"),
        {"schema": "news"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Summary period
    symbol = Column(String(20), nullable=False)
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    
    # Article counts
    total_articles = Column(Integer, nullable=False)
    positive_articles = Column(Integer, nullable=False)
    negative_articles = Column(Integer, nullable=False)
    neutral_articles = Column(Integer, nullable=False)
    
    # Aggregated metrics
    avg_sentiment = Column(Float, nullable=False)  # -1 to 1
    sentiment_std = Column(Float, nullable=False)
    
    # Volume metrics
    article_velocity = Column(Float, nullable=False)  # Articles per day
    velocity_change = Column(Float, nullable=False)  # Change from previous period
    
    # Key topics
    top_categories = Column(JSON)
    key_entities = Column(JSON)
    
    # Notable articles
    most_positive_id = Column(UUID(as_uuid=True))
    most_negative_id = Column(UUID(as_uuid=True))
    most_impactful_id = Column(UUID(as_uuid=True))
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)