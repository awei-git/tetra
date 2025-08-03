from typing import List, Set, Dict
from fastapi import WebSocket
import logging
import json
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[str, Set[WebSocket]] = {
            "coverage": set(),
            "query": set(),
            "all": set()
        }
    
    async def connect(self, websocket: WebSocket):
        """Accept and register new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions["all"].add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.remove(websocket)
        for topic in self.subscriptions:
            self.subscriptions[topic].discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific connection"""
        await websocket.send_text(message)
    
    async def send_personal_json(self, data: dict, websocket: WebSocket):
        """Send JSON data to specific connection"""
        await websocket.send_json(data)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connections"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {str(e)}")
    
    async def broadcast_json(self, data: dict):
        """Broadcast JSON data to all connections"""
        message = json.dumps(data)
        await self.broadcast(message)
    
    async def broadcast_to_topic(self, topic: str, data: dict):
        """Broadcast to subscribers of a specific topic"""
        if topic not in self.subscriptions:
            return
        
        message = json.dumps({
            "topic": topic,
            "data": data
        })
        
        for connection in self.subscriptions[topic]:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to topic {topic}: {str(e)}")
    
    async def subscribe(self, websocket: WebSocket, topic: str):
        """Subscribe connection to a topic"""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
        self.subscriptions[topic].add(websocket)
    
    async def unsubscribe(self, websocket: WebSocket, topic: str):
        """Unsubscribe connection from a topic"""
        if topic in self.subscriptions:
            self.subscriptions[topic].discard(websocket)


# Global connection manager instance
connection_manager = ConnectionManager()


# Background task for heartbeat
async def heartbeat_task():
    """Send periodic heartbeat to keep connections alive"""
    while True:
        await asyncio.sleep(30)  # Every 30 seconds
        await connection_manager.broadcast_json({
            "type": "heartbeat",
            "timestamp": datetime.utcnow().isoformat()
        })