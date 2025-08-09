#!/bin/bash

# Remove old aliases
unalias tetra-logs tetra-restart tetra-webgui 2>/dev/null

# Set correct aliases
alias tetra="/Users/angwei/Repos/tetra/bin/launch_services.sh"
alias tetra-stop="/Users/angwei/Repos/tetra/bin/stop_services.sh"
alias tetra-restart="tetra-stop && sleep 2 && tetra"
alias tetra-logs="tail -f /tmp/tetra-backend.log /tmp/tetra-frontend.log"
alias tetra-backend-log="tail -f /tmp/tetra-backend.log"
alias tetra-frontend-log="tail -f /tmp/tetra-frontend.log"
alias tetra-status="ps aux | grep -E '(uvicorn|npm)' | grep -v grep"
alias tetra-web="open http://localhost:3000"

echo "Tetra aliases have been fixed!"
echo "Available commands:"
echo "  tetra         - Start all services"
echo "  tetra-stop    - Stop all services"
echo "  tetra-restart - Restart all services"
echo "  tetra-logs    - View logs"
echo "  tetra-status  - Check service status"
echo "  tetra-web     - Open web interface"