#!/bin/bash
# Script to restart WebGUI backend and frontend
# This will be called by cron at 6 AM daily

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/webgui_restart_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "========================================="
log "Starting WebGUI restart process"
log "========================================="

# Kill existing processes
log "Stopping existing processes..."

# Find and kill backend process (uvicorn)
BACKEND_PID=$(pgrep -f "uvicorn app.main:app")
if [ ! -z "$BACKEND_PID" ]; then
    log "Killing backend process (PID: $BACKEND_PID)"
    kill $BACKEND_PID
    sleep 2
fi

# Find and kill frontend process (vite)
FRONTEND_PID=$(pgrep -f "vite.*frontend")
if [ ! -z "$FRONTEND_PID" ]; then
    log "Killing frontend process (PID: $FRONTEND_PID)"
    kill $FRONTEND_PID
    sleep 2
fi

# Start backend
log "Starting backend..."
cd "$PROJECT_ROOT/backend"

# Check if virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "$PROJECT_ROOT/../dev" ]; then
    source "$PROJECT_ROOT/../dev/bin/activate"
fi

# Start backend in background
nohup uvicorn app.main:app --reload --port 8000 --host 0.0.0.0 > "$PROJECT_ROOT/logs/backend_$(date +%Y%m%d).log" 2>&1 &
BACKEND_NEW_PID=$!
log "Backend started with PID: $BACKEND_NEW_PID"

# Give backend time to start
sleep 5

# Start frontend
log "Starting frontend..."
cd "$PROJECT_ROOT/frontend"

# Start frontend in background
nohup npm run dev > "$PROJECT_ROOT/logs/frontend_$(date +%Y%m%d).log" 2>&1 &
FRONTEND_NEW_PID=$!
log "Frontend started with PID: $FRONTEND_NEW_PID"

# Verify processes are running
sleep 5
if ps -p $BACKEND_NEW_PID > /dev/null; then
    log "✅ Backend is running"
else
    log "❌ Backend failed to start"
fi

if ps -p $FRONTEND_NEW_PID > /dev/null; then
    log "✅ Frontend is running"
else
    log "❌ Frontend failed to start"
fi

log "========================================="
log "WebGUI restart completed"
log "========================================="

# Clean up old logs (keep last 7 days)
find "$PROJECT_ROOT/logs" -name "webgui_restart_*.log" -mtime +7 -delete
find "$PROJECT_ROOT/logs" -name "backend_*.log" -mtime +7 -delete
find "$PROJECT_ROOT/logs" -name "frontend_*.log" -mtime +7 -delete