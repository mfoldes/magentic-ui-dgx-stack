#!/bin/bash
# ============================================================================
# validate-deployment.sh - Validate Complete Deployment
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

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║           Magentic-UI Deployment Validation                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

passed=0
failed=0

run_test() {
    local name="$1"
    local cmd="$2"
    
    echo -n "Testing: $name... "
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        ((passed++))
    else
        echo -e "${RED}FAILED${NC}"
        ((failed++))
    fi
}

echo ""

# Infrastructure tests
echo -e "${BLUE}Infrastructure Tests:${NC}"
run_test "NVIDIA GPU available" "nvidia-smi"
run_test "Docker running" "docker ps"
run_test "Python 3 available" "python3 --version"

echo ""

# vLLM tests
echo -e "${BLUE}vLLM Server Tests:${NC}"
run_test "vLLM container running" "docker ps | grep -q vllm-fara"
run_test "vLLM health endpoint" "curl -s http://localhost:${VLLM_PORT}/health"
run_test "vLLM models endpoint" "curl -s http://localhost:${VLLM_PORT}/v1/models"

echo ""

# Inference test
echo -e "${BLUE}Inference Test:${NC}"
echo -n "Testing: vLLM inference... "
INFERENCE_RESULT=$(curl -s "http://localhost:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "microsoft/Fara-7B",
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 10,
        "temperature": 0
    }' 2>/dev/null || echo "")

if echo "$INFERENCE_RESULT" | grep -q "choices"; then
    echo -e "${GREEN}PASSED${NC}"
    ((passed++))
    
    # Extract response
    RESPONSE=$(echo "$INFERENCE_RESULT" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:50])" 2>/dev/null || echo "")
    if [ -n "$RESPONSE" ]; then
        echo "  Response: $RESPONSE"
    fi
else
    echo -e "${RED}FAILED${NC}"
    ((failed++))
fi

echo ""

# Environment tests
echo -e "${BLUE}Environment Tests:${NC}"
run_test "Virtual environment exists" "[ -d '${PROJECT_ROOT}/venv' ]"
run_test ".env file exists" "[ -f '${PROJECT_ROOT}/.env' ]"
run_test "Config directory exists" "[ -d '${PROJECT_ROOT}/config' ]"

echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "Results: ${GREEN}${passed} passed${NC}, ${RED}${failed} failed${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $failed -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Some tests failed. Check the output above for details.${NC}"
    exit 1
else
    echo ""
    echo -e "${GREEN}All tests passed! Deployment is ready.${NC}"
    exit 0
fi
