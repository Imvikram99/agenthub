#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────
# start_all.sh — Launch all Agent Hub Services
# IMPORTANT: OpenClaw also needs to be running for this ecosystem to fully work.
# (Start it separately via `openclaw gateway`)
# ────────────────────────────────────────────────────────────────
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/services"
PID_FILE="$PROJECT_ROOT/logs/.service_pids"
mkdir -p "$LOG_DIR"

echo -e "${CYAN}🚀 Starting Agent Hub Infrastructure...${NC}"

# 1. Start Background Docker Services (Redis + Qdrant)
echo -e "${YELLOW}Starting Docker Services (Redis, Qdrant)...${NC}"
# Setup a lightweight compose or manual docker run if needed (assuming they are running on host for now, or start via docker)
if ! docker ps --format '{{.Names}}' | grep -q '^redis-hub$'; then
    docker rm redis-hub 2>/dev/null || true; docker run -d --name redis-hub -p 6379:6379 redis:alpine || echo "Redis already running"
fi
if ! docker ps --format '{{.Names}}' | grep -q '^qdrant-hub$'; then
    docker rm qdrant-hub 2>/dev/null || true; docker run -d --name qdrant-hub -p 6333:6333 qdrant/qdrant || echo "Qdrant already running"
fi

# 2. Start LangGraph POC Service
echo -e "${YELLOW}Starting langgraph-poc backend...${NC}"
cd "$PROJECT_ROOT/../langgraph-poc"
source .venv/bin/activate
nohup uvicorn src.ui.server:app --port 8000 > "$LOG_DIR/langgraph_poc.log" 2>&1 &
LANGGRAPH_PID=$!
echo -e "${GREEN}✓ langgraph-poc started (PID: $LANGGRAPH_PID)${NC}"

# 3. Start Salesy Backend Service
echo -e "${YELLOW}Starting salesy backend...${NC}"
cd "$PROJECT_ROOT/../salesy/backend"
source ../.venv/bin/activate
nohup uvicorn app.main:app --port 8001 > "$LOG_DIR/salesy.log" 2>&1 &
SALESY_PID=$!
echo -e "${GREEN}✓ salesy backend started (PID: $SALESY_PID)${NC}"

# 4. Start Salesy Frontend Service
echo -e "${YELLOW}Starting salesy frontend...${NC}"
cd "$PROJECT_ROOT/../salesy/frontend"
# Check if node_modules exists, if not run npm install
if [ ! -d "node_modules" ]; then
    echo -e "${CYAN}Installing frontend dependencies...${NC}"
    npm install > "$LOG_DIR/salesy_frontend_install.log" 2>&1
fi
nohup npm start > "$LOG_DIR/salesy_frontend.log" 2>&1 &
SALESY_FE_PID=$!
echo -e "${GREEN}✓ salesy frontend started (PID: $SALESY_FE_PID)${NC}"

# 5. Start Hub Gateway
echo -e "${YELLOW}Starting Agent Hub Gateway...${NC}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
nohup python -m app.gateway --port 9000 > "$LOG_DIR/hub_gateway.log" 2>&1 &
HUB_PID=$!
echo -e "${GREEN}✓ Agent Hub Gateway started (PID: $HUB_PID)${NC}"

# 6. Start ARQ Worker (for CLI commands like rothchild)
echo -e "${YELLOW}Starting ARQ Background Worker...${NC}"
nohup arq app.hub.worker.WorkerSettings > "$LOG_DIR/hub_arq_worker.log" 2>&1 &
ARQ_PID=$!
echo -e "${GREEN}✓ ARQ worker started (PID: $ARQ_PID)${NC}"

# 7. Start OpenClaw Gateway
echo -e "${YELLOW}Starting OpenClaw Gateway...${NC}"
nohup openclaw gateway --port 18789 > "$LOG_DIR/openclaw_gateway.log" 2>&1 &
OPENCLAW_PID=$!
echo -e "${GREEN}✓ OpenClaw Gateway started (PID: $OPENCLAW_PID)${NC}"

# 8. Start Telegram Bot Receiver
echo -e "${YELLOW}Starting Telegram Bot Receiver...${NC}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
nohup python -m app.telegram_receiver > "$LOG_DIR/telegram_receiver.log" 2>&1 &
TELEGRAM_PID=$!
echo -e "${GREEN}✓ Telegram Bot Receiver started (PID: $TELEGRAM_PID)${NC}"

# Save PIDs for stop_all.sh
echo "$LANGGRAPH_PID $SALESY_PID $SALESY_FE_PID $HUB_PID $ARQ_PID $OPENCLAW_PID $TELEGRAM_PID" > "$PID_FILE"

echo -e "\n${CYAN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}All services are running!${NC}"
echo -e "Logs available in $LOG_DIR/"
echo -e "PIDs saved to $PID_FILE"
echo -e "To stop: ./scripts/stop_all.sh"
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
