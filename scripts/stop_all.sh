#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────
# stop_all.sh — Stop all Agent Hub Services
# Cleanly shuts down all services started by start_all.sh.
# ────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🛑 Stopping Agent Hub Infrastructure...${NC}"

# Kill processes by name/pattern
declare -a PATTERNS=(
    "app.gateway"
    "app.telegram_receiver"
    "app.hub.worker"
    "openclaw gateway"
    "hub_cli.py"
    "run_antigravity.py"
    "src.ui.server"       # langgraph-poc
    "salesy.*app.main"    # salesy backend
)

killed=0
for pattern in "${PATTERNS[@]}"; do
    pids=$(pgrep -f "$pattern" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo -e "  Killing ${pattern}... (PIDs: $pids)"
        echo "$pids" | xargs kill 2>/dev/null
        killed=$((killed + ${#pids[@]}))
    fi
done

# Stop Docker containers
echo -e "${CYAN}Stopping Docker containers...${NC}"
docker stop redis-hub 2>/dev/null && docker rm redis-hub 2>/dev/null && echo -e "  ${GREEN}✓ redis-hub stopped${NC}" || true
docker stop qdrant-hub 2>/dev/null && docker rm qdrant-hub 2>/dev/null && echo -e "  ${GREEN}✓ qdrant-hub stopped${NC}" || true

# Clean up lock files
rm -f /tmp/telegram_receiver.lock 2>/dev/null

echo -e "\n${GREEN}All Agent Hub services stopped.${NC}"
