# Fine-Tuning Fara-7B with Unsloth on DGX Spark

This directory contains tools and configurations for fine-tuning Microsoft's Fara-7B model using Unsloth for memory-efficient LoRA training on NVIDIA DGX Spark.

## Overview

[Unsloth](https://github.com/unslothai/unsloth) provides 2-5x faster training with 60% less memory usage compared to standard fine-tuning, making it ideal for the DGX Spark's unified memory architecture.

### Why Fine-Tune Fara-7B?

Fara-7B is already trained for computer use tasks, but fine-tuning can:
- Adapt to specific applications or domains
- Improve performance on your particular workflows
- Add custom actions or behaviors
- Reduce inference steps for common tasks

## Prerequisites

- DGX Spark with NVIDIA Blackwell GB10
- Python 3.10+
- CUDA 12.x
- ~25-40GB GPU memory for training (easily available on Spark's 128GB)

## Quick Start

### 1. Install Dependencies

```bash
cd fine-tuning
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

```bash
python prepare-dataset.py \
    --input /path/to/your/data \
    --output ./data/training.jsonl \
    --format fara
```

### 3. Configure Training

Edit `configs/training-config.yaml` to adjust:
- LoRA rank and alpha
- Learning rate
- Batch size
- Number of epochs

### 4. Run Training

```bash
python train-unsloth.py --config configs/training-config.yaml
```

### 5. Merge Adapters (Optional)

To create a standalone fine-tuned model:

```bash
python merge-adapter.py \
    --adapter ./output/checkpoint-final \
    --output ./merged-model
```

## Dataset Format

### Fara-7B Trajectory Format

For Computer Use Agent (CUA) tasks:

```json
{
    "task": "Navigate to google.com and search for 'AI news'",
    "trajectory": [
        {
            "observation": "<base64_screenshot>",
            "thought": "I need to navigate to Google first",
            "action": "visit_url(https://google.com)"
        },
        {
            "observation": "<base64_screenshot>",
            "thought": "Now I'll type the search query",
            "action": "type('AI news')"
        },
        {
            "observation": "<base64_screenshot>",
            "thought": "I'll click the search button",
            "action": "click(640, 450)"
        }
    ]
}
```

### Chat/Conversation Format

For simpler instruction-following:

```json
{
    "messages": [
        {"role": "user", "content": "Click on the login button"},
        {"role": "assistant", "content": "Thought: I need to find and click the login button\nAction: click(1200, 50)"}
    ]
}
```

## Memory Requirements

| Configuration | GPU Memory | Training Speed | Accuracy |
|--------------|------------|----------------|----------|
| LoRA r=8 | ~20 GB | Fastest | Good |
| LoRA r=16 | ~25 GB | Fast | Better |
| LoRA r=32 | ~35 GB | Medium | Best |
| LoRA r=64 | ~50 GB | Slower | Maximum |

DGX Spark's 128GB unified memory comfortably supports all configurations.

## Configuration Options

### Training Configuration (`configs/training-config.yaml`)

```yaml
model:
  base_model: "microsoft/Fara-7B"
  max_seq_length: 16384
  load_in_4bit: false  # Full precision on Spark

lora:
  r: 16                # LoRA rank
  alpha: 32            # LoRA alpha (typically 2x rank)
  dropout: 0.05        # Dropout for regularization

training:
  epochs: 3
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2e-4
```

### LoRA Presets (`configs/lora-config.yaml`)

Pre-configured settings for different use cases:
- `quick`: Fast training, good results
- `balanced`: Recommended default
- `quality`: Best results, slower training
- `max`: Maximum quality, most memory

## Files

| File | Description |
|------|-------------|
| `train-unsloth.py` | Main training script |
| `prepare-dataset.py` | Dataset preparation utility |
| `merge-adapter.py` | Merge LoRA adapters into base model |
| `requirements.txt` | Python dependencies |
| `configs/training-config.yaml` | Training hyperparameters |
| `configs/lora-config.yaml` | LoRA preset configurations |

## Tips for DGX Spark

### 1. Clear Memory Cache Before Training

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

### 2. Monitor GPU Memory

```bash
watch -n 1 nvidia-smi
```

### 3. Use Gradient Checkpointing

Already enabled by default in the training script for longer sequences.

### 4. Start with Smaller LoRA Rank

Begin with r=16 and increase if you need better quality.

### 5. Use BF16 Precision

The training script automatically uses BF16 on Blackwell architecture for optimal performance.

## Evaluation

After training, evaluate your model:

```bash
# Start vLLM with your fine-tuned model
vllm serve ./output/checkpoint-final --port 5000

# Or if you merged the adapter
vllm serve ./merged-model --port 5000

# Then use the standard validation script
../scripts/validate-deployment.sh
```

## Troubleshooting

### Out of Memory

- Reduce batch size
- Reduce LoRA rank
- Enable gradient checkpointing (default)
- Clear memory cache before training

### Slow Training

- Increase batch size if memory allows
- Use BF16 precision (default on Spark)
- Reduce max_seq_length if your data doesn't need it

### Poor Results

- Train for more epochs
- Increase LoRA rank
- Increase learning rate slightly
- Check dataset quality

## Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Fara-7B Model Card](https://huggingface.co/microsoft/Fara-7B)
- [Fara-7B Paper](https://arxiv.org/abs/2511.19663)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
