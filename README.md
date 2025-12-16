# Magentic-UI DGX Spark Stack

A complete deployment stack for running Magentic-UI with Microsoft's Fara-7B model on NVIDIA DGX Spark.

## Overview

This stack provides an end-to-end solution for deploying Magentic-UI, a human-in-the-loop agentic system built on AutoGen, using Fara-7B as the vision-language model for computer use tasks. The deployment is optimized for the DGX Spark's Blackwell GB10 architecture.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User's Web Browser                               │
│                        http://localhost:4200                             │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Magentic-UI (Direct Python)                           │
│            (AutoGen Team Manager + Agent Team)                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐         │
│  │ Orchestrator│ │  WebSurfer  │ │    Coder    │ │ FileSurfer │         │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └─────┬──────┘         │
│         └───────────────┴───────────────┴──────────────┘                 │
│                                  │                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │              Chromium Browser (Playwright)                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ OpenAI-compatible API
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     vLLM Inference Server (Docker)                       │
│                    microsoft/Fara-7B @ localhost:5000                    │
│              (GB10 GPU • FP16/Auto • 128GB Unified Memory)              │
└─────────────────────────────────────────────────────────────────────────┘
```

## Hardware Specifications

### DGX Spark

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA Blackwell GB10 |
| CUDA Cores | 6,144 |
| Tensor Cores | 5th Generation with FP4 support |
| Memory | 128 GB unified LPDDR5x |
| Memory Bandwidth | 273 GB/s |
| CPU | 20-core ARM64 (10x Cortex-X925 + 10x Cortex-A725) |
| AI Compute | Up to 1,000 TOPS inference, 1 PFLOP @ FP4 |

### Model Memory Requirements

| Precision | Fara-7B Size | Remaining for System | Recommended For |
|-----------|--------------|----------------------|-----------------|
| FP16/Auto | ~14 GB | ~114 GB | Development (Best Accuracy) |
| FP8 | ~7 GB | ~121 GB | Balanced |
| NVFP4 | ~3.5 GB | ~124.5 GB | Memory Constrained |

**Recommendation:** Use FP16 or auto precision for optimal accuracy. The DGX Spark's 128GB unified memory provides ample headroom.

## Quick Start

### Prerequisites

1. DGX Spark with DGX OS installed
2. Docker and NVIDIA Container Runtime configured
3. Python 3.10+ installed
4. HuggingFace account with access to microsoft/Fara-7B

### Installation

```bash
# 1. Navigate to this directory
cd /path/to/magentic-ui-dgx-stack

# 2. Run the complete setup
./scripts/setup-complete.sh

# 3. Start vLLM server with Fara-7B
./scripts/start-vllm.sh

# 4. In a new terminal, start Magentic-UI
./scripts/start-magentic-ui.sh

# 5. Access Magentic-UI at http://localhost:4200
```

### Verify Installation

```bash
# Check all components
./scripts/health-check.sh

# Run validation tests
./scripts/validate-deployment.sh

# Run performance benchmarks
./scripts/benchmark.sh
```

## Directory Structure

```
magentic-ui-dgx-stack/
├── README.md                    # This file
├── .env.template                # Environment template (copy to .env)
├── scripts/                     # Automation scripts
│   ├── setup-complete.sh        # Complete end-to-end setup
│   ├── start-vllm.sh            # Start vLLM server with Fara-7B
│   ├── start-magentic-ui.sh     # Start Magentic-UI
│   ├── stop-services.sh         # Stop all services
│   ├── health-check.sh          # Service health checks
│   ├── validate-deployment.sh   # Validate installation
│   └── benchmark.sh             # Performance benchmarks
├── config/                      # Configuration files
│   └── fara-config.yaml         # FARA-7B model client configuration
├── fine-tuning/                 # Fine-tuning resources
│   ├── README.md                # Fine-tuning documentation
│   ├── requirements.txt         # Python dependencies
│   ├── train-unsloth.py         # Unsloth training script
│   ├── prepare-dataset.py       # Dataset preparation
│   ├── merge-adapter.py         # Merge LoRA adapters
│   └── configs/                 # Training configurations
│       ├── training-config.yaml # Training hyperparameters
│       └── lora-config.yaml     # LoRA adapter settings
├── docker/                      # Docker configurations (vLLM only)
│   └── docker-compose.yml       # vLLM container composition
├── models/                      # Model storage
├── workspace/                   # Shared workspace for agents
├── logs/                        # Log files
└── benchmark-results/           # Benchmark outputs
```

## Configuration

### Environment Variables

Copy `.env.template` to `.env` and configure:

```bash
cp .env.template .env
# Edit .env with your settings
```

Key variables:
- `HUGGINGFACE_TOKEN` - Required for model access
- `VLLM_PORT` - vLLM server port (default: 5000)
- `MAGENTIC_UI_PORT` - Magentic-UI port (default: 4200)
- `VLLM_DTYPE` - Precision (auto, float16, float8)

## Fine-Tuning with Unsloth

For custom fine-tuning to improve computer use effectiveness:

```bash
cd fine-tuning

# 1. Install fine-tuning dependencies
pip install -r requirements.txt

# 2. Prepare your dataset
python prepare-dataset.py --input /path/to/data --output ./data/training.jsonl

# 3. Run training
python train-unsloth.py --config configs/training-config.yaml

# 4. Merge adapters (optional)
python merge-adapter.py --adapter ./output/checkpoint-final --output ./merged-model
```

See `fine-tuning/README.md` for detailed documentation.

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Clear cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| Model loading fails | Verify HuggingFace token and model access |
| vLLM connection refused | Check vLLM server: `./scripts/health-check.sh` |
| Browser issues | Reinstall Playwright: `playwright install chromium` |

### View Logs

```bash
# vLLM logs
docker logs -f vllm-fara

# Magentic-UI logs (in terminal where it's running)
# Or check logs/ directory
```

## Resources

- [Magentic-UI GitHub](https://github.com/microsoft/magentic-ui)
- [Magentic-UI Paper](https://arxiv.org/abs/2507.22358)
- [Fara-7B on HuggingFace](https://huggingface.co/microsoft/Fara-7B)
- [Fara-7B Paper](https://arxiv.org/abs/2511.19663)
- [vLLM Documentation](https://docs.vllm.ai)
- [DGX Spark Documentation](https://docs.nvidia.com/dgx/)
- [Unsloth Fine-Tuning](https://github.com/unslothai/unsloth)

## License

This deployment stack is provided for development and research purposes. Please refer to the individual licenses for Magentic-UI, Fara-7B, and vLLM.
