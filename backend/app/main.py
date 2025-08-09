from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .config import settings
from .routers import monitor, chat, strategies
from .services.database import init_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up WebGUI backend...")
    try:
        await init_db()
        logger.info("Database connection established")
    except Exception as e:
        logger.warning(f"Database connection failed: {e}")
        logger.warning("Running without database - only LLM features will work")
    yield
    # Shutdown
    logger.info("Shutting down WebGUI backend...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(monitor.router, prefix="/api/monitor", tags=["monitor"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(strategies.router, prefix="/api/strategies", tags=["strategies"])

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.app_name,
        "llm_provider": settings.llm_provider
    }

# WebSocket endpoint for real-time updates
from fastapi import WebSocket, WebSocketDisconnect
from .services.websocket import connection_manager

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for now
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)