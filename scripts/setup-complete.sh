#!/bin/bash
# ============================================================================
# setup-complete.sh - Complete End-to-End Setup for Magentic-UI on DGX Spark
# ============================================================================
# This script performs a complete installation of:
#   - vLLM inference server with Fara-7B model (Docker)
#   - Magentic-UI multi-agent system (direct Python install)
#   - Required dependencies and configurations
#
# Target: NVIDIA DGX Spark (Blackwell GB10, 128GB Unified Memory)
# ============================================================================

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
LOG_FILE="${PROJECT_ROOT}/logs/setup-$(date +%Y%m%d-%H%M%S).log"

# Default settings
VLLM_PORT="${VLLM_PORT:-5000}"
MAGENTIC_UI_PORT="${MAGENTIC_UI_PORT:-4200}"
FARA_MODEL="${FARA_MODEL:-microsoft/Fara-7B}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"

# ============================================================================
# Utility Functions
# ============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        INFO)  echo -e "${BLUE}[INFO]${NC} ${message}" ;;
        OK)    echo -e "${GREEN}[OK]${NC} ${message}" ;;
        WARN)  echo -e "${YELLOW}[WARN]${NC} ${message}" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} ${message}" ;;
        STEP)  echo -e "${CYAN}[STEP]${NC} ${message}" ;;
    esac
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                       ║"
    echo "║        Magentic-UI + Fara-7B on DGX Spark                            ║"
    echo "║        Complete Setup Script                                          ║"
    echo "║                                                                       ║"
    echo "╚═══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_section() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN} $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

check_prerequisites() {
    print_section "Checking Prerequisites"
    
    local errors=0
    
    # Check GPU
    log STEP "Checking NVIDIA GPU..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        log OK "NVIDIA GPU detected"
    else
        log ERROR "nvidia-smi not found. Is NVIDIA driver installed?"
        ((errors++))
    fi
    
    # Check Docker
    log STEP "Checking Docker..."
    if command -v docker &> /dev/null; then
        docker --version
        log OK "Docker installed"
    else
        log ERROR "Docker not found. Please install Docker."
        ((errors++))
    fi
    
    # Check NVIDIA Container Toolkit
    log STEP "Checking NVIDIA Container Toolkit..."
    if docker info 2>/dev/null | grep -q "nvidia"; then
        log OK "NVIDIA Container Toolkit configured"
    else
        log WARN "NVIDIA Container Toolkit may not be configured"
    fi
    
    # Check Python
    log STEP "Checking Python..."
    if command -v python3 &> /dev/null; then
        python3 --version
        log OK "Python3 installed"
    else
        log ERROR "Python3 not found. Please install Python 3.10+."
        ((errors++))
    fi
    
    # Check pip
    log STEP "Checking pip..."
    if command -v pip3 &> /dev/null; then
        log OK "pip3 installed"
    else
        log WARN "pip3 not found. Will attempt to install."
    fi
    
    # Check available memory
    log STEP "Checking system memory..."
    local total_mem=$(free -g | awk '/^Mem:/{print $2}')
    log INFO "Total system memory: ${total_mem}GB"
    
    # Check disk space
    log STEP "Checking disk space..."
    local free_space=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    log INFO "Free disk space: ${free_space}GB"
    
    if [ $errors -gt 0 ]; then
        log ERROR "Prerequisites check failed with $errors errors"
        return 1
    fi
    
    log OK "All prerequisites satisfied"
    return 0
}

# ============================================================================
# Environment Setup
# ============================================================================

setup_environment() {
    print_section "Setting Up Environment"
    
    # Create required directories
    log STEP "Creating directory structure..."
    mkdir -p "${PROJECT_ROOT}/models"
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/workspace"
    mkdir -p "${PROJECT_ROOT}/benchmark-results"
    mkdir -p "${PROJECT_ROOT}/config"
    mkdir -p "${PROJECT_ROOT}/docker"
    mkdir -p "${PROJECT_ROOT}/fine-tuning/configs"
    mkdir -p "${PROJECT_ROOT}/fine-tuning/data"
    mkdir -p "${PROJECT_ROOT}/fine-tuning/output"
    log OK "Directories created"
    
    # Setup .env file if not exists
    if [ ! -f "${PROJECT_ROOT}/.env" ]; then
        if [ -f "${PROJECT_ROOT}/.env.template" ]; then
            log STEP "Creating .env from template..."
            cp "${PROJECT_ROOT}/.env.template" "${PROJECT_ROOT}/.env"
            log OK "Environment file created at ${PROJECT_ROOT}/.env"
            log WARN "Please edit .env and add your HUGGINGFACE_TOKEN"
        else
            log STEP "Creating environment file..."
            cat > "${PROJECT_ROOT}/.env" << 'ENVEOF'
# Magentic-UI DGX Spark Stack Environment Configuration

# HuggingFace Configuration (REQUIRED)
HUGGINGFACE_TOKEN=

# vLLM Server Configuration
VLLM_HOST=localhost
VLLM_PORT=5000
VLLM_MODEL=microsoft/Fara-7B
VLLM_DTYPE=auto
VLLM_MAX_MODEL_LEN=16384
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_IMAGE=nvcr.io/nvidia/vllm:25.09-py3

# Magentic-UI Configuration
MAGENTIC_UI_PORT=4200
MAGENTIC_UI_LOG_LEVEL=INFO

# GPU Settings
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
ENVEOF
            log OK "Environment file created"
            log WARN "Please edit .env and add your HUGGINGFACE_TOKEN"
        fi
    else
        log OK "Environment file already exists"
    fi
    
    # Source environment
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        set -a
        source "${PROJECT_ROOT}/.env"
        set +a
    fi
}

# ============================================================================
# HuggingFace Setup
# ============================================================================

setup_huggingface() {
    print_section "Setting Up HuggingFace"
    
    # Install HuggingFace CLI if needed
    if ! command -v huggingface-cli &> /dev/null; then
        log STEP "Installing HuggingFace Hub..."
        pip3 install --user huggingface_hub[cli]
        log OK "HuggingFace Hub installed"
    else
        log OK "HuggingFace CLI already installed"
    fi
    
    # Check if token is set
    if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
        log WARN "HUGGINGFACE_TOKEN not set in environment"
        echo ""
        read -p "Enter your HuggingFace token (or press Enter to skip): " hf_token
        if [ -n "$hf_token" ]; then
            export HUGGINGFACE_TOKEN="$hf_token"
            sed -i "s/^HUGGINGFACE_TOKEN=.*/HUGGINGFACE_TOKEN=$hf_token/" "${PROJECT_ROOT}/.env" 2>/dev/null || \
            sed -i '' "s/^HUGGINGFACE_TOKEN=.*/HUGGINGFACE_TOKEN=$hf_token/" "${PROJECT_ROOT}/.env"
            log OK "HuggingFace token saved"
        else
            log WARN "Skipping HuggingFace login. Model download may fail."
            return 0
        fi
    fi
    
    # Login to HuggingFace
    log STEP "Logging into HuggingFace..."
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential 2>/dev/null || true
    log OK "HuggingFace login complete"
}

# ============================================================================
# vLLM Setup
# ============================================================================

setup_vllm() {
    print_section "Setting Up vLLM with Fara-7B"
    
    local VLLM_IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:25.09-py3}"
    
    # Pull NVIDIA vLLM image
    log STEP "Pulling NVIDIA vLLM image..."
    docker pull "$VLLM_IMAGE"
    log OK "vLLM image pulled: $VLLM_IMAGE"
    
    # Pre-download the model (optional, speeds up first start)
    log STEP "Model will be downloaded on first vLLM start"
    log INFO "Model: ${FARA_MODEL}"
    log INFO "This may take several minutes on first run"
}

# ============================================================================
# Magentic-UI Setup
# ============================================================================

setup_magentic_ui() {
    print_section "Setting Up Magentic-UI"
    
    # Create Python virtual environment
    local VENV_DIR="${PROJECT_ROOT}/venv"
    
    if [ ! -d "$VENV_DIR" ]; then
        log STEP "Creating Python virtual environment..."
        python3 -m venv "$VENV_DIR"
        log OK "Virtual environment created"
    else
        log OK "Virtual environment already exists"
    fi
    
    # Activate and install dependencies
    log STEP "Installing Magentic-UI dependencies..."
    source "${VENV_DIR}/bin/activate"
    
    pip install --upgrade pip setuptools wheel
    
    # Install magentic-ui from GitHub
    pip install "magentic-ui[all] @ git+https://github.com/microsoft/magentic-ui.git"
    
    # Install Playwright browsers
    log STEP "Installing Playwright browsers..."
    playwright install chromium
    
    deactivate
    
    log OK "Magentic-UI installed successfully"
    
    # Create endpoint configuration
    log STEP "Creating endpoint configuration..."
    cat > "${PROJECT_ROOT}/config/endpoint-config.json" << EOF
{
    "model": "${FARA_MODEL}",
    "base_url": "http://localhost:${VLLM_PORT}/v1",
    "api_key": "local-vllm",
    "max_tokens": 2048,
    "temperature": 0.0
}
EOF
    log OK "Endpoint configuration created"
}

# ============================================================================
# Create Startup Scripts
# ============================================================================

create_scripts() {
    print_section "Creating Startup Scripts"
    
    # Make all scripts executable
    chmod +x "${SCRIPT_DIR}"/*.sh 2>/dev/null || true
    
    log OK "All scripts are ready"
}

# ============================================================================
# Final Summary
# ============================================================================

print_summary() {
    print_section "Setup Complete"
    
    echo -e "${GREEN}✓ All components installed successfully!${NC}"
    echo ""
    echo "Directory: ${PROJECT_ROOT}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Ensure your HUGGINGFACE_TOKEN is set in .env"
    echo ""
    echo "  2. Start vLLM server with Fara-7B:"
    echo "     ${SCRIPT_DIR}/start-vllm.sh"
    echo ""
    echo "  3. In a new terminal, start Magentic-UI:"
    echo "     ${SCRIPT_DIR}/start-magentic-ui.sh"
    echo ""
    echo "  4. Access Magentic-UI at:"
    echo "     http://localhost:${MAGENTIC_UI_PORT}"
    echo ""
    echo "Other commands:"
    echo "  - Health check:  ${SCRIPT_DIR}/health-check.sh"
    echo "  - Stop services: ${SCRIPT_DIR}/stop-services.sh"
    echo "  - Benchmarks:    ${SCRIPT_DIR}/benchmark.sh"
    echo ""
    echo "For fine-tuning, see: ${PROJECT_ROOT}/fine-tuning/README.md"
    echo ""
}

# ============================================================================
# Main
# ============================================================================

main() {
    mkdir -p "$(dirname "$LOG_FILE")"
    
    print_banner
    
    log INFO "Starting Magentic-UI DGX Spark Stack setup"
    log INFO "Log file: $LOG_FILE"
    
    check_prerequisites || { log ERROR "Prerequisites check failed"; exit 1; }
    setup_environment
    setup_huggingface
    setup_vllm
    setup_magentic_ui
    create_scripts
    print_summary
    
    log OK "Setup completed successfully"
}

main "$@"
