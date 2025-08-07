#!/bin/bash
# Launch Tetra backend and frontend services
# Scheduled to run daily at 5:00 AM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[$(date)] Starting Tetra services..."

# Function to check if process is running
is_running() {
    pgrep -f "$1" > /dev/null 2>&1
}

# Kill existing processes
echo "Stopping any existing services..."
pkill -f "uvicorn backend.main:app" || true
pkill -f "npm.*run dev.*webgui" || true
pkill -f "next-server" || true

# Wait for processes to stop
sleep 2

# Start backend
echo "Starting backend service..."
cd "$PROJECT_ROOT"
source .venv/bin/activate
nohup python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/tetra-backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend started successfully"
        break
    fi
    sleep 1
done

# Start frontend
echo "Starting frontend service..."
cd "$PROJECT_ROOT/webgui"
nohup npm run dev > /tmp/tetra-frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to start
echo "Waiting for frontend to start..."
for i in {1..30}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "Frontend started successfully"
        break
    fi
    sleep 1
done

echo "[$(date)] Tetra services launched successfully"
echo "Backend PID: $BACKEND_PID (logs: /tmp/tetra-backend.log)"
echo "Frontend PID: $FRONTEND_PID (logs: /tmp/tetra-frontend.log)"