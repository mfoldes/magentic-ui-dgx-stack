#!/bin/bash
# ============================================================================
# stop-services.sh - Stop All Magentic-UI Stack Services
# ============================================================================

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Stopping Magentic-UI Stack services...${NC}"
echo ""

# Stop vLLM container
if docker ps --format '{{.Names}}' | grep -q "^vllm-fara$"; then
    echo "Stopping vLLM container..."
    docker stop vllm-fara
    docker rm vllm-fara
    echo -e "${GREEN}✓ vLLM container stopped${NC}"
else
    echo -e "${YELLOW}vLLM container not running${NC}"
fi

echo ""
echo -e "${GREEN}All services stopped.${NC}"
echo ""
echo "Note: If Magentic-UI is running in another terminal, press Ctrl+C to stop it."
