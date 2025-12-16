#!/bin/bash
# ============================================================================
# start-vllm.sh - Start vLLM Server with Fara-7B
# ============================================================================
# Runs vLLM in a Docker container with Fara-7B model
# Target: NVIDIA DGX Spark (Blackwell GB10, 128GB Unified Memory)
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

# Configuration with defaults
VLLM_IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:25.09-py3}"
VLLM_PORT="${VLLM_PORT:-5000}"
VLLM_MODEL="${VLLM_MODEL:-microsoft/Fara-7B}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║              Starting vLLM Server with Fara-7B                        ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}Configuration:${NC}"
echo "  Model:              $VLLM_MODEL"
echo "  Port:               $VLLM_PORT"
echo "  Dtype:              $VLLM_DTYPE"
echo "  Max Model Length:   $VLLM_MAX_MODEL_LEN"
echo "  GPU Memory Util:    $VLLM_GPU_MEMORY_UTILIZATION"
echo ""

# Check if container already running
if docker ps --format '{{.Names}}' | grep -q "^vllm-fara$"; then
    echo -e "${YELLOW}vLLM container already running${NC}"
    echo "Use './scripts/stop-services.sh' to stop it first"
    exit 0
fi

# Remove stopped container if exists
docker rm -f vllm-fara 2>/dev/null || true

# Clear memory cache (recommended for DGX Spark)
echo -e "${BLUE}Clearing memory cache...${NC}"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true

# Check HuggingFace token
if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
    echo -e "${YELLOW}Warning: HUGGINGFACE_TOKEN not set${NC}"
    echo "Model download may fail. Set it in .env file."
    echo ""
fi

echo -e "${BLUE}Starting vLLM container...${NC}"
echo ""

# Run vLLM container
docker run -d \
    --name vllm-fara \
    --gpus all \
    --runtime nvidia \
    -p ${VLLM_PORT}:${VLLM_PORT} \
    -v "${PROJECT_ROOT}/models:/models" \
    -e HF_TOKEN="${HUGGINGFACE_TOKEN:-}" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    --restart unless-stopped \
    ${VLLM_IMAGE} \
    vllm serve "${VLLM_MODEL}" \
        --host 0.0.0.0 \
        --port ${VLLM_PORT} \
        --dtype ${VLLM_DTYPE} \
        --max-model-len ${VLLM_MAX_MODEL_LEN} \
        --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION} \
        --trust-remote-code

echo -e "${GREEN}vLLM container started${NC}"
echo ""
echo "Container: vllm-fara"
echo "API endpoint: http://localhost:${VLLM_PORT}"
echo ""
echo -e "${BLUE}Waiting for server to be ready (this may take a few minutes for first run)...${NC}"

# Wait for server to be ready
for i in {1..120}; do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}✓ vLLM server is ready!${NC}"
        echo ""
        echo "API endpoint: http://localhost:${VLLM_PORT}"
        echo "OpenAI-compatible: http://localhost:${VLLM_PORT}/v1"
        echo ""
        echo "Test with:"
        echo "  curl http://localhost:${VLLM_PORT}/v1/models"
        echo ""
        echo "View logs:"
        echo "  docker logs -f vllm-fara"
        exit 0
    fi
    echo -n "."
    sleep 5
done

echo ""
echo -e "${YELLOW}Server did not respond within timeout${NC}"
echo "Check logs: docker logs vllm-fara"
exit 1
