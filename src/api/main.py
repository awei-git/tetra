from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings
from src.api.routers import market_data, events, event_data, health
from src.utils.logging import setup_logging


# Setup logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    logger.info("Starting Tetra Trading Platform API")
    # Startup
    yield
    # Shutdown
    logger.info("Shutting down Tetra Trading Platform API")


# Create FastAPI app
app = FastAPI(
    title="Tetra Trading Platform",
    description="Comprehensive quantitative trading platform API",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
app.include_router(market_data.router, prefix="/api/v1/market-data", tags=["market-data"])
app.include_router(events.router, prefix="/api/v1/events", tags=["events"])  # Old events API (to be deprecated)
app.include_router(event_data.router, prefix="/api/v1/event-data", tags=["event-data"])  # New event data API


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Tetra Trading Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }