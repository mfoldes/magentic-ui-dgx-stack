#!/bin/bash
# ============================================================================
# start-magentic-ui.sh - Start Magentic-UI
# ============================================================================
# Runs Magentic-UI directly using Python with the --port flag
# Target: NVIDIA DGX Spark
# ============================================================================

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# Configuration
MAGENTIC_UI_HOST="${MAGENTIC_UI_HOST:-0.0.0.0}"
MAGENTIC_UI_PORT="${MAGENTIC_UI_PORT:-4200}"
VLLM_PORT="${VLLM_PORT:-5000}"
VLLM_MODEL="${VLLM_MODEL:-microsoft/Fara-7B}"
VENV_DIR="${PROJECT_ROOT}/venv"
FARA_CONFIG="${PROJECT_ROOT}/config/fara-config.yaml"

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                     Starting Magentic-UI                              ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}Configuration:${NC}"
echo "  Host:           $MAGENTIC_UI_HOST"
echo "  Port:           $MAGENTIC_UI_PORT"
echo "  vLLM Endpoint:  http://localhost:${VLLM_PORT}"
echo "  Model:          $VLLM_MODEL"
echo "  Config:         $FARA_CONFIG"
echo ""

# Check if vLLM is running
echo -e "${BLUE}Checking vLLM server...${NC}"
if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ vLLM server is running${NC}"
else
    echo -e "${RED}✗ vLLM server is not running${NC}"
    echo ""
    echo "Please start vLLM first:"
    echo "  ${SCRIPT_DIR}/start-vllm.sh"
    echo ""
    exit 1
fi
echo ""

# Check for virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv "$VENV_DIR"
    
    echo "Installing Magentic-UI..."
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip setuptools wheel
    # Install from PyPI (includes pre-built frontend)
    # Note: Installing from git does NOT include frontend build files
    pip install "magentic-ui[all]"
    playwright install chromium
    deactivate
    echo -e "${GREEN}✓ Magentic-UI installed${NC}"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Set environment variables for Magentic-UI
#export OPENAI_API_KEY="local-vllm"
#export OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1"
#export MODEL_NAME="${VLLM_MODEL}"

# Create workspace directory if needed
mkdir -p "${PROJECT_ROOT}/workspace"

echo -e "${GREEN}Starting Magentic-UI...${NC}"
echo ""
echo "Access at: http://localhost:${MAGENTIC_UI_PORT}"
echo ""
echo "Press Ctrl+C to stop"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run Magentic-UI with host/port flags and FARA config
# Binding to 0.0.0.0 enables live browser view and network access
cd "${PROJECT_ROOT}/workspace"
magentic-ui --fara --host ${MAGENTIC_UI_HOST} --port ${MAGENTIC_UI_PORT} --config "${FARA_CONFIG}"
