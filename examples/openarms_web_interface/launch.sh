#!/bin/bash

# OpenArms Web Interface Launcher
# Starts Rerun viewer, FastAPI backend, and React frontend

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   OpenArms Web Recording Interface    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down services...${NC}"
    
    # Kill all child processes
    pkill -P $$ 2>/dev/null || true
    
    # Kill specific services by port
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true  # Backend
    lsof -ti:5173 | xargs kill -9 2>/dev/null || true  # Frontend
    lsof -ti:9876 | xargs kill -9 2>/dev/null || true  # Rerun (if spawned)
    
    echo -e "${GREEN}✓ Services stopped${NC}"
    exit 0
}

# Register cleanup on script exit
trap cleanup EXIT INT TERM

# Check if required commands exist
command -v rerun >/dev/null 2>&1 || { 
    echo -e "${RED}✗ Error: 'rerun' not found. Please install: pip install rerun-sdk${NC}"
    exit 1
}

command -v python >/dev/null 2>&1 || { 
    echo -e "${RED}✗ Error: 'python' not found${NC}"
    exit 1
}

command -v npm >/dev/null 2>&1 || { 
    echo -e "${RED}✗ Error: 'npm' not found${NC}"
    exit 1
}

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}⚠ node_modules not found. Running npm install...${NC}"
    npm install
    echo -e "${GREEN}✓ Dependencies installed${NC}"
    echo ""
fi

echo -e "${GREEN}Starting services...${NC}"
echo ""

# 1. Start FastAPI backend (Rerun will start when recording begins)
echo -e "${BLUE}[1/2]${NC} Starting FastAPI backend on port 8000..."
cd "$SCRIPT_DIR"
python web_record_server.py > /tmp/openarms_backend.log 2>&1 &
BACKEND_PID=$!
sleep 3

if ps -p $BACKEND_PID > /dev/null; then
    echo -e "${GREEN}✓ Backend started${NC} (PID: $BACKEND_PID)"
    echo -e "      URL: ${BLUE}http://localhost:8000${NC}"
else
    echo -e "${RED}✗ Failed to start backend${NC}"
    echo -e "${YELLOW}Check logs: tail -f /tmp/openarms_backend.log${NC}"
    exit 1
fi
echo ""

# 2. Start React frontend
echo -e "${BLUE}[2/2]${NC} Starting React frontend on port 5173..."
cd "$SCRIPT_DIR"
npm run dev > /tmp/openarms_frontend.log 2>&1 &
FRONTEND_PID=$!
sleep 3

if ps -p $FRONTEND_PID > /dev/null; then
    echo -e "${GREEN}✓ Frontend started${NC} (PID: $FRONTEND_PID)"
    echo -e "      URL: ${BLUE}http://localhost:5173${NC}"
else
    echo -e "${RED}✗ Failed to start frontend${NC}"
    echo -e "${YELLOW}Check logs: tail -f /tmp/openarms_frontend.log${NC}"
    exit 1
fi
echo ""

# Display status
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     All services running! 🚀           ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "🔧 ${BLUE}Backend:${NC}  http://localhost:8000"
echo -e "🌐 ${BLUE}Frontend:${NC} http://localhost:5173"
echo -e "📊 ${BLUE}Rerun:${NC}    Will spawn automatically when recording starts"
echo ""
echo -e "${YELLOW}Open your browser to:${NC} ${BLUE}http://localhost:5173${NC}"
echo ""
echo -e "${YELLOW}Logs:${NC}"
echo -e "  • Backend:  tail -f /tmp/openarms_backend.log"
echo -e "  • Frontend: tail -f /tmp/openarms_frontend.log"
echo ""
echo -e "${RED}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep script running and wait for any service to exit
wait

