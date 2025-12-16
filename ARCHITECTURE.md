# Magentic-UI DGX Spark Stack - Architecture & Technical Guide

This document provides a comprehensive technical overview of the Magentic-UI DGX Spark Stack, including architecture, workflows, configuration, and potential improvements.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Architecture & How It Works](#architecture--how-it-works)
- [Workflow: End-to-End User Experience](#workflow-end-to-end-user-experience)
- [Data Flow](#data-flow)
- [Configuration Files Explained](#configuration-files-explained)
- [Fine-Tuning Workflow](#fine-tuning-workflow)
- [Potential Improvements](#potential-improvements)
- [Summary](#summary)

---

## Repository Structure

```
magentic-ui-dgx-stack/
├── README.md                          # Main documentation
├── ARCHITECTURE.md                    # This file
├── .env.template                      # Environment configuration template
│
├── scripts/                           # 8 automation scripts
│   ├── setup-complete.sh              # Full setup (creates venv, installs deps)
│   ├── start-vllm.sh                  # Starts Fara-7B in Docker container
│   ├── start-magentic-ui.sh           # Installs/runs Magentic-UI natively
│   ├── start-services.sh              # Orchestrates startup sequence
│   ├── stop-services.sh               # Stops all services
│   ├── health-check.sh                # Comprehensive health diagnostics
│   ├── validate-deployment.sh         # Automated test suite
│   └── benchmark.sh                   # Performance benchmarking
│
├── config/                            # 3 configuration files
│   ├── vllm-config.yaml               # vLLM server settings (reference)
│   ├── magentic-ui-config.yaml        # Agent configuration (reference)
│   └── endpoint-config.json           # API endpoint for Magentic-UI
│
├── docker/
│   └── docker-compose.yml             # vLLM container only
│
├── fine-tuning/                       # Complete Unsloth training toolkit
│   ├── README.md                      # Comprehensive fine-tuning guide
│   ├── requirements.txt               # Python dependencies
│   ├── train-unsloth.py               # Main training script
│   ├── prepare-dataset.py             # Dataset formatter
│   ├── merge-adapter.py               # LoRA adapter merger
│   ├── configs/
│   │   ├── training-config.yaml       # Training hyperparameters
│   │   └── lora-config.yaml           # LoRA presets
│   ├── data/                          # Training data directory
│   └── output/                        # Training outputs
│
├── models/                            # Model storage (optional local caching)
├── workspace/                         # Shared agent workspace
├── logs/                              # Log files
└── benchmark-results/                 # Performance test results
```

---

## Architecture & How It Works

### System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              USER'S BROWSER                                   │
│                          http://localhost:4200                                │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      │ WebSocket + HTTP
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    MAGENTIC-UI (Native Python on Host)                        │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         AUTOGEN TEAM MANAGER                           │ │
│  │  Manages sessions, routes messages, stores chat history (SQLite)       │ │
│  └───────────────────────────────────┬────────────────────────────────────┘ │
│                                      │                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           AGENT TEAM                                   │ │
│  │                                                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │ │
│  │  │ ORCHESTRATOR │  │  WEBSURFER   │  │    CODER     │  │ FILESURFER │ │ │
│  │  │              │  │              │  │              │  │            │ │ │
│  │  │ • Plans tasks│  │ • Browser    │  │ • Generates  │  │ • File I/O │ │ │
│  │  │ • Assigns    │  │   automation │  │   code       │  │ • Converts │ │ │
│  │  │   agents     │  │ • Screenshots│  │ • Executes   │  │   formats  │ │ │
│  │  │ • Tracks     │  │ • Click/type │  │   in sandbox │  │            │ │ │
│  │  │   progress   │  │              │  │              │  │            │ │ │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │ │
│  │         │                 │                 │                │        │ │
│  └─────────┼─────────────────┼─────────────────┼────────────────┼────────┘ │
│            │                 │                 │                │          │
│            │    ┌────────────▼────────────┐    │                │          │
│            │    │  PLAYWRIGHT CHROMIUM    │    │                │          │
│            │    │  (Docker sandboxed)     │    │                │          │
│            │    └─────────────────────────┘    │                │          │
│            │                                   │                │          │
│            └───────────────┬───────────────────┼────────────────┘          │
│                            │                   │                           │
│                            ▼                   ▼                           │
│                    ┌───────────────┐   ┌───────────────┐                   │
│                    │   workspace/  │   │ Code Sandbox  │                   │
│                    │  (shared dir) │   │   (Docker)    │                   │
│                    └───────────────┘   └───────────────┘                   │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │ OpenAI-compatible API
                              │ POST /v1/chat/completions
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    vLLM INFERENCE SERVER (Docker Container)                   │
│                                                                              │
│    Container: vllm-fara                                                      │
│    Image: nvcr.io/nvidia/vllm:25.09-py3                                     │
│    Port: 5000 (configurable)                                                 │
│                                                                              │
│    ┌──────────────────────────────────────────────────────────────────────┐ │
│    │                     FARA-7B MODEL                                    │ │
│    │                                                                      │ │
│    │  • Vision-Language Model (7B parameters)                            │ │
│    │  • Based on Qwen2.5-VL-7B                                           │ │
│    │  • Trained for Computer Use Agent (CUA) tasks                       │ │
│    │  • Predicts coordinates directly from screenshots                   │ │
│    │  • Average ~16 steps per task                                       │ │
│    └──────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         DGX SPARK HARDWARE                                    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    BLACKWELL GB10 GPU                                   ││
│  │  • 6,144 CUDA Cores                                                     ││
│  │  • 5th Gen Tensor Cores (FP4 support)                                   ││
│  │  • 128 GB Unified LPDDR5x Memory                                        ││
│  │  • 273 GB/s Memory Bandwidth                                            ││
│  │  • Up to 1,000 TOPS inference                                           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    ARM64 CPU                                            ││
│  │  • 20 cores (10x Cortex-X925 + 10x Cortex-A725)                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Workflow: End-to-End User Experience

### Initial Setup (One Time)

```bash
# 1. Clone/navigate to the repo
cd /path/to/magentic-ui-dgx-stack

# 2. Copy and configure environment
cp .env.template .env
# Edit .env → Add HUGGINGFACE_TOKEN

# 3. Run complete setup
./scripts/setup-complete.sh
```

**What `setup-complete.sh` does:**

1. Checks prerequisites (GPU, Docker, Python)
2. Creates directory structure
3. Sets up HuggingFace authentication
4. Creates `.env` file if missing
5. Creates Python virtual environment
6. Installs Magentic-UI in the venv
7. Installs Playwright browsers

### Daily Operation

**Terminal 1: Start vLLM**

```bash
./scripts/start-vllm.sh
```

- Clears memory cache
- Pulls NVIDIA vLLM container if needed
- Starts Fara-7B model
- Waits for health check (up to 10 minutes for first run)
- Exposes API at `http://localhost:5000`

**Terminal 2: Start Magentic-UI**

```bash
./scripts/start-magentic-ui.sh
```

- Activates virtual environment
- Sets `OPENAI_BASE_URL=http://localhost:5000/v1`
- Runs `magentic-ui --port 4200`
- User accesses `http://localhost:4200`

**Stopping**

```bash
./scripts/stop-services.sh    # Stops vLLM container
# Ctrl+C in Terminal 2         # Stops Magentic-UI
```

### Using Magentic-UI

1. Open `http://localhost:4200`
2. Enter a task (e.g., "Find the cheapest flight from NYC to LA next week")
3. Magentic-UI creates a plan and shows it for approval (co-planning)
4. Approve/modify the plan
5. Watch agents execute:
   - **Orchestrator** delegates steps
   - **WebSurfer** navigates websites, takes screenshots
   - **Coder** writes scripts if needed
   - **FileSurfer** handles file operations
6. Review results and provide feedback

---

## Data Flow

```
User Input → Orchestrator → Creates Plan → User Approves
                                              ↓
                              Plan Execution Loop
                                              ↓
         ┌────────────────────────────────────┴───────────────────────────────┐
         ↓                        ↓                        ↓                  ↓
    WebSurfer              Coder Agent            FileSurfer           UserProxy
         │                        │                        │                  │
         ↓                        ↓                        ↓                  ↓
    Screenshot              Code Block              File Path            User Input
    + Context               + Context              + Context             + Context
         │                        │                        │                  │
         └────────────────────────┴────────────────────────┴──────────────────┘
                                              ↓
                                         vLLM API
                                              ↓
                                    Fara-7B Inference
                                    (Vision + Language)
                                              ↓
                                    Action Prediction
                                    (click, type, scroll, etc.)
                                              ↓
                                    Agent Executes Action
                                              ↓
                                    Result → Orchestrator
                                              ↓
                                    Next Step or Complete
```

---

## Configuration Files Explained

| File | Purpose | Used By |
|------|---------|---------|
| `.env` | Runtime environment variables | All scripts |
| `config/vllm-config.yaml` | Reference documentation for vLLM settings | Human reference |
| `config/magentic-ui-config.yaml` | Reference for agent configuration | Human reference |
| `config/endpoint-config.json` | Can be passed to Magentic-UI for API settings | `magentic-ui` CLI |

**Note:** The YAML configs are primarily documentation/reference. The actual runtime configuration comes from:

- `.env` file (loaded by scripts)
- Command-line arguments to `vllm serve` and `magentic-ui`
- Environment variables (`OPENAI_BASE_URL`, etc.)

---

## Fine-Tuning Workflow

```
Raw Data                    Formatted Data                 Trained Model
─────────────────────────────────────────────────────────────────────────────
Your trajectories  →  prepare-dataset.py  →  data/training.jsonl
                                                      │
                                                      ▼
                                            train-unsloth.py
                                            (LoRA training)
                                                      │
                                                      ▼
                                          output/checkpoint-final
                                                      │
                              ┌───────────────────────┴────────────────────────┐
                              ↓                                                ↓
                      Use as Adapter                              merge-adapter.py
                      (load on top of Fara-7B)                           │
                                                                         ▼
                                                              merged-model/
                                                              (standalone)
```

### Fine-Tuning Steps

1. **Prepare Dataset**
   ```bash
   cd fine-tuning
   python prepare-dataset.py --input /path/to/data --output ./data/training.jsonl
   ```

2. **Configure Training**
   Edit `configs/training-config.yaml` to set LoRA rank, learning rate, epochs, etc.

3. **Run Training**
   ```bash
   python train-unsloth.py --config configs/training-config.yaml
   ```

4. **Merge Adapters (Optional)**
   ```bash
   python merge-adapter.py --adapter ./output/checkpoint-final --output ./merged-model
   ```

5. **Deploy Fine-Tuned Model**
   Update `.env` to point to your fine-tuned model, or modify the vLLM serve command.

---

## Potential Improvements

### High Priority

| Improvement | Rationale | Effort |
|-------------|-----------|--------|
| **Add `.env` file creation in setup** | `setup-complete.sh` references `.env` but doesn't copy from template | Low |
| **systemd service files** | Enable auto-start on boot for production use | Medium |
| **TLS/HTTPS support** | Security for network deployments | Medium |
| **Log rotation** | Prevent disk filling in long-running deployments | Low |

### Medium Priority

| Improvement | Rationale | Effort |
|-------------|-----------|--------|
| **Prometheus metrics** | Observability for vLLM performance | Medium |
| **Backup script** | Save workspace, configs, fine-tuned models | Low |
| **Multi-model support** | Allow switching between Fara-7B and other models | Medium |
| **GPU monitoring dashboard** | Real-time visualization of GPU memory/utilization | Medium |

### Nice-to-Have

| Improvement | Rationale | Effort |
|-------------|-----------|--------|
| **Integration tests** | Automated end-to-end testing of agent workflows | High |
| **Model quantization scripts** | Easy switching between FP16/FP8/INT4 | Medium |
| **Docker Compose override** | Allow users to customize without editing main compose | Low |
| **VS Code devcontainer** | Streamlined development experience | Medium |

### Documentation Improvements

| Improvement | Rationale |
|-------------|-----------|
| Add troubleshooting for ARM64-specific issues | DGX Spark is ARM64-based |
| Document memory tuning for different workloads | Help users optimize |
| Add example fine-tuning datasets | Lower barrier to customization |
| Create video walkthrough | Visual learners |

---

## Summary

This repository is a **well-structured, production-ready deployment stack** that:

1. **Cleanly separates concerns**: vLLM runs in Docker, Magentic-UI runs natively
2. **Follows DGX Spark best practices**: Memory cache clearing, appropriate precision settings, NGC container usage
3. **Provides complete automation**: Setup, start, stop, health check, validation, benchmarking
4. **Includes fine-tuning capability**: Full Unsloth toolkit for model customization
5. **Is well-documented**: Clear README, inline comments, reference configurations

The architecture leverages the DGX Spark's 128GB unified memory to run Fara-7B at full FP16 precision (no accuracy loss from quantization), while the multi-agent Magentic-UI system provides human-in-the-loop oversight for complex agentic tasks.

---

## Quick Reference

### Ports

| Service | Default Port | Environment Variable |
|---------|--------------|---------------------|
| vLLM API | 5000 | `VLLM_PORT` |
| Magentic-UI | 4200 | `MAGENTIC_UI_PORT` |

### Key Commands

```bash
# Setup
./scripts/setup-complete.sh

# Start
./scripts/start-vllm.sh          # Terminal 1
./scripts/start-magentic-ui.sh   # Terminal 2

# Monitor
./scripts/health-check.sh
./scripts/validate-deployment.sh
./scripts/benchmark.sh

# Stop
./scripts/stop-services.sh       # Stops vLLM
# Ctrl+C                          # Stops Magentic-UI

# Logs
docker logs -f vllm-fara
```

### Memory Requirements

| Component | Memory Usage |
|-----------|--------------|
| Fara-7B (FP16) | ~14 GB |
| vLLM KV Cache | ~10-30 GB (dynamic) |
| Magentic-UI | ~2-4 GB |
| Playwright Browser | ~1-2 GB |
| **Total** | **~30-50 GB** |
| **Available on Spark** | **128 GB** |

---

*Document generated for magentic-ui-dgx-stack repository*
