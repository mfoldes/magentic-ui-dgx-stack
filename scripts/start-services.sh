#!/bin/bash
# ============================================================================
# start-services.sh - Start All Magentic-UI Stack Services
# ============================================================================
# Starts vLLM in Docker, then provides instructions for Magentic-UI
# ============================================================================

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

VLLM_PORT="${VLLM_PORT:-5000}"
MAGENTIC_UI_PORT="${MAGENTIC_UI_PORT:-4200}"

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║              Starting Magentic-UI Stack Services                      ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Clear memory cache
echo -e "${BLUE}Clearing memory cache...${NC}"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true

# Start vLLM
echo -e "\n${BLUE}Starting vLLM server...${NC}"
"${SCRIPT_DIR}/start-vllm.sh"

# Instructions for Magentic-UI
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}vLLM is running!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "vLLM API: http://localhost:${VLLM_PORT}"
echo ""
echo -e "${YELLOW}To start Magentic-UI, open a new terminal and run:${NC}"
echo ""
echo "    ${SCRIPT_DIR}/start-magentic-ui.sh"
echo ""
echo "Then access Magentic-UI at: http://localhost:${MAGENTIC_UI_PORT}"
echo ""
echo -e "${BLUE}Other commands:${NC}"
echo "  Health check:  ${SCRIPT_DIR}/health-check.sh"
echo "  Stop services: ${SCRIPT_DIR}/stop-services.sh"
echo "  View vLLM logs: docker logs -f vllm-fara"
echo ""
