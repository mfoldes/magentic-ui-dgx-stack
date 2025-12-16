#!/bin/bash
# ============================================================================
# benchmark.sh - Performance Benchmark for Fara-7B on DGX Spark
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
VLLM_MODEL="${VLLM_MODEL:-microsoft/Fara-7B}"
RESULTS_DIR="${PROJECT_ROOT}/benchmark-results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/benchmark-${TIMESTAMP}.json"

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║           Fara-7B Performance Benchmark                               ║"
echo "║           DGX Spark (Blackwell GB10)                                  ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

mkdir -p "$RESULTS_DIR"

# Check vLLM is running
echo -e "${BLUE}Checking vLLM server...${NC}"
if ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "Error: vLLM server not running"
    echo "Start it with: ./scripts/start-vllm.sh"
    exit 1
fi
echo -e "${GREEN}✓ vLLM server is running${NC}"
echo ""

# Get GPU info
echo -e "${BLUE}GPU Information:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo ""

echo -e "${BLUE}Running benchmarks...${NC}"
echo ""

# ============================================================================
# Benchmark 1: Short completion (64 tokens)
# ============================================================================
echo -e "${CYAN}▶ Short Completion (64 tokens):${NC}"
short_times=()
for i in {1..5}; do
    start_time=$(date +%s.%N)
    curl -s "http://localhost:${VLLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${VLLM_MODEL}"'",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 64,
            "temperature": 0
        }' > /dev/null
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)
    short_times+=("$elapsed")
    echo -n "."
done
echo ""
short_avg=$(echo "${short_times[@]}" | tr ' ' '\n' | awk '{sum+=$1} END {printf "%.3f", sum/NR}')
echo "  Average: ${short_avg}s per request"

# ============================================================================
# Benchmark 2: Medium completion (256 tokens)
# ============================================================================
echo ""
echo -e "${CYAN}▶ Medium Completion (256 tokens):${NC}"
medium_times=()
for i in {1..3}; do
    start_time=$(date +%s.%N)
    curl -s "http://localhost:${VLLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${VLLM_MODEL}"'",
            "messages": [{"role": "user", "content": "Explain how neural networks learn in a paragraph."}],
            "max_tokens": 256,
            "temperature": 0
        }' > /dev/null
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)
    medium_times+=("$elapsed")
    echo -n "."
done
echo ""
medium_avg=$(echo "${medium_times[@]}" | tr ' ' '\n' | awk '{sum+=$1} END {printf "%.3f", sum/NR}')
echo "  Average: ${medium_avg}s per request"

# ============================================================================
# Benchmark 3: Long completion (512 tokens)
# ============================================================================
echo ""
echo -e "${CYAN}▶ Long Completion (512 tokens):${NC}"
long_times=()
for i in {1..3}; do
    start_time=$(date +%s.%N)
    curl -s "http://localhost:${VLLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${VLLM_MODEL}"'",
            "messages": [{"role": "user", "content": "Write a detailed guide on setting up a Python web application with best practices."}],
            "max_tokens": 512,
            "temperature": 0
        }' > /dev/null
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)
    long_times+=("$elapsed")
    echo -n "."
done
echo ""
long_avg=$(echo "${long_times[@]}" | tr ' ' '\n' | awk '{sum+=$1} END {printf "%.3f", sum/NR}')
echo "  Average: ${long_avg}s per request"

# ============================================================================
# Save Results
# ============================================================================
echo ""
echo -e "${BLUE}Saving results...${NC}"

cat > "$RESULTS_FILE" << EOF
{
    "timestamp": "$TIMESTAMP",
    "platform": "DGX Spark",
    "gpu": "Blackwell GB10",
    "model": "${VLLM_MODEL}",
    "results": {
        "short_completion_64_tokens": {
            "iterations": 5,
            "avg_time_sec": $short_avg
        },
        "medium_completion_256_tokens": {
            "iterations": 3,
            "avg_time_sec": $medium_avg
        },
        "long_completion_512_tokens": {
            "iterations": 3,
            "avg_time_sec": $long_avg
        }
    }
}
EOF

echo -e "${GREEN}Results saved to: $RESULTS_FILE${NC}"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${CYAN}Benchmark Summary:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Short (64 tokens):   ${short_avg}s"
echo "  Medium (256 tokens): ${medium_avg}s"
echo "  Long (512 tokens):   ${long_avg}s"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# GPU memory after benchmarks
echo ""
echo -e "${BLUE}GPU Memory Usage:${NC}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
