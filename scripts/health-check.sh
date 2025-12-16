#!/bin/bash
# ============================================================================
# health-check.sh - Check Health of All Services
# ============================================================================

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
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
echo "║              Magentic-UI Stack Health Check                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ============================================================================
# GPU Status
# ============================================================================
echo -e "${BLUE}▶ GPU Status:${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  GPU query failed"
else
    echo -e "  ${RED}✗ nvidia-smi not available${NC}"
fi
echo ""

# ============================================================================
# Docker Containers
# ============================================================================
echo -e "${BLUE}▶ Docker Containers:${NC}"
if docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null | grep -E "(vllm|magentic)" ; then
    echo ""
else
    echo "  No Magentic-UI stack containers running"
fi
echo ""

# ============================================================================
# vLLM Server
# ============================================================================
echo -e "${BLUE}▶ vLLM Server (port $VLLM_PORT):${NC}"
if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓ vLLM server is healthy${NC}"
    
    # Get model info
    MODEL_INFO=$(curl -s "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null)
    if [ -n "$MODEL_INFO" ]; then
        MODEL_NAME=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "unknown")
        echo "  Model: $MODEL_NAME"
    fi
else
    echo -e "  ${RED}✗ vLLM server not responding${NC}"
    
    # Check if container exists but not healthy
    if docker ps -a --format '{{.Names}}' | grep -q "^vllm-fara$"; then
        CONTAINER_STATUS=$(docker ps -a --format '{{.Status}}' --filter "name=vllm-fara")
        echo "  Container status: $CONTAINER_STATUS"
    fi
fi
echo ""

# ============================================================================
# Magentic-UI
# ============================================================================
echo -e "${BLUE}▶ Magentic-UI (port $MAGENTIC_UI_PORT):${NC}"
if curl -s "http://localhost:${MAGENTIC_UI_PORT}" > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓ Magentic-UI is running${NC}"
else
    echo -e "  ${YELLOW}○ Magentic-UI not responding (may not be started)${NC}"
fi
echo ""

# ============================================================================
# Virtual Environment
# ============================================================================
echo -e "${BLUE}▶ Python Environment:${NC}"
VENV_DIR="${PROJECT_ROOT}/venv"
if [ -d "$VENV_DIR" ]; then
    echo -e "  ${GREEN}✓ Virtual environment exists${NC}"
    if [ -f "${VENV_DIR}/bin/magentic-ui" ] || "${VENV_DIR}/bin/pip" show magentic-ui &>/dev/null; then
        echo -e "  ${GREEN}✓ Magentic-UI installed${NC}"
    else
        echo -e "  ${YELLOW}○ Magentic-UI not installed in venv${NC}"
    fi
else
    echo -e "  ${YELLOW}○ Virtual environment not created${NC}"
fi
echo ""

# ============================================================================
# System Resources
# ============================================================================
echo -e "${BLUE}▶ System Resources:${NC}"
echo "  Memory: $(free -h | awk '/^Mem:/{print $3 "/" $2}')"
echo "  Disk:   $(df -h "${PROJECT_ROOT}" | awk 'NR==2{print $3 "/" $2 " (" $5 " used)"}')"
echo ""

# ============================================================================
# Configuration
# ============================================================================
echo -e "${BLUE}▶ Configuration:${NC}"
if [ -f "${PROJECT_ROOT}/.env" ]; then
    echo -e "  ${GREEN}✓ .env file exists${NC}"
    if grep -q "^HUGGINGFACE_TOKEN=.\+" "${PROJECT_ROOT}/.env" 2>/dev/null; then
        echo -e "  ${GREEN}✓ HuggingFace token configured${NC}"
    else
        echo -e "  ${YELLOW}○ HuggingFace token not set${NC}"
    fi
else
    echo -e "  ${RED}✗ .env file missing${NC}"
fi
echo ""
