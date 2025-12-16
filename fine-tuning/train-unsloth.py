#!/usr/bin/env python3
"""
train-unsloth.py - Fine-tune Fara-7B using Unsloth for memory-efficient LoRA training

This script provides memory-efficient fine-tuning of Microsoft's Fara-7B model
using Unsloth optimizations, targeting NVIDIA DGX Spark with Blackwell GB10.

Usage:
    python train-unsloth.py --config configs/training-config.yaml
    python train-unsloth.py --config configs/training-config.yaml --resume ./output/checkpoint-500

Author: Magentic-UI DGX Stack
Target: NVIDIA DGX Spark (Blackwell GB10, 128GB Unified Memory)
"""

import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path

import yaml
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def print_banner():
    """Print startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║           Fara-7B Fine-Tuning with Unsloth                            ║
║           Target: NVIDIA DGX Spark (Blackwell GB10)                   ║
╚═══════════════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="cyan"))


def check_gpu():
    """Check GPU availability and print info."""
    if not torch.cuda.is_available():
        console.print("[red]Error: CUDA not available. GPU required for training.[/red]")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    console.print(f"[green]✓ GPU detected:[/green] {gpu_name}")
    console.print(f"[green]✓ GPU Memory:[/green] {gpu_memory:.1f} GB")
    
    return gpu_name, gpu_memory


def format_training_sample(sample: dict) -> str:
    """Format a training sample for the model."""
    
    # Handle trajectory-based format
    if 'trajectory' in sample:
        parts = [f"Task: {sample.get('task', 'Complete the following task')}\n"]
        for step in sample['trajectory']:
            thought = step.get('thought', '')
            action = step.get('action', '')
            parts.append(f"Thought: {thought}")
            parts.append(f"Action: {action}")
        return "\n".join(parts)
    
    # Handle chat/messages format
    elif 'messages' in sample:
        parts = []
        for msg in sample['messages']:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            parts.append(f"{role}: {content}")
        return "\n".join(parts)
    
    # Handle simple text format
    elif 'text' in sample:
        return sample['text']
    
    else:
        raise ValueError(f"Unknown sample format: {list(sample.keys())}")


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune Fara-7B with Unsloth on DGX Spark',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to training configuration YAML file'
    )
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None, 
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without training'
    )
    args = parser.parse_args()
    
    print_banner()
    
    # Load configuration
    console.print(f"\n[blue]Loading configuration:[/blue] {args.config}")
    config = load_config(args.config)
    
    # Print configuration summary
    console.print("\n[cyan]Configuration:[/cyan]")
    console.print(f"  Base model:     {config['model']['base_model']}")
    console.print(f"  Max seq length: {config['model'].get('max_seq_length', 16384)}")
    console.print(f"  LoRA rank:      {config['lora']['r']}")
    console.print(f"  LoRA alpha:     {config['lora']['alpha']}")
    console.print(f"  Epochs:         {config['training']['epochs']}")
    console.print(f"  Batch size:     {config['training']['batch_size']}")
    console.print(f"  Learning rate:  {config['training']['learning_rate']}")
    
    # Check GPU
    console.print("\n[blue]Checking GPU...[/blue]")
    gpu_name, gpu_memory = check_gpu()
    
    if args.dry_run:
        console.print("\n[yellow]Dry run complete. Configuration is valid.[/yellow]")
        return 0
    
    # Import Unsloth (after GPU check)
    console.print("\n[blue]Loading Unsloth...[/blue]")
    try:
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
        console.print("[green]✓ Unsloth loaded successfully[/green]")
    except ImportError as e:
        console.print(f"[red]Error: Unsloth not installed: {e}[/red]")
        console.print("Install with: pip install -r requirements.txt")
        return 1
    
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from datasets import load_dataset
    
    # Extract configuration
    model_config = config['model']
    lora_config = config['lora']
    train_config = config['training']
    output_config = config.get('output', {'dir': './output', 'save_steps': 500})
    data_config = config.get('data', {'path': './data/training.jsonl'})
    
    # Load model with Unsloth optimizations
    console.print("\n[blue]Loading model...[/blue]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading Fara-7B...", total=None)
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config['base_model'],
            max_seq_length=model_config.get('max_seq_length', 16384),
            dtype=None,  # Auto-detect best dtype
            load_in_4bit=model_config.get('load_in_4bit', False),
        )
        progress.update(task, completed=True)
    
    console.print("[green]✓ Model loaded successfully[/green]")
    
    # Add LoRA adapters
    console.print("\n[blue]Adding LoRA adapters...[/blue]")
    
    target_modules = lora_config.get('target_modules', [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('alpha', 32),
        lora_dropout=lora_config.get('dropout', 0.05),
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=model_config.get('max_seq_length', 16384),
    )
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]✓ LoRA adapters added[/green]")
    console.print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Load dataset
    console.print("\n[blue]Loading dataset...[/blue]")
    data_path = data_config.get('path', './data/training.jsonl')
    
    if os.path.exists(data_path):
        dataset = load_dataset('json', data_files=data_path, split='train')
        console.print(f"[green]✓ Dataset loaded:[/green] {len(dataset)} samples")
    else:
        console.print(f"[yellow]Warning: Dataset not found at {data_path}[/yellow]")
        console.print("Creating minimal test dataset...")
        
        # Create minimal test data
        test_data = [
            {"text": "Task: Click the search button\nThought: I need to click the search button\nAction: click(640, 400)"},
            {"text": "Task: Type hello\nThought: I need to type hello in the input field\nAction: type('hello')"},
        ]
        os.makedirs(os.path.dirname(data_path) or '.', exist_ok=True)
        with open(data_path, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        dataset = load_dataset('json', data_files=data_path, split='train')
        console.print(f"[yellow]Created test dataset with {len(dataset)} samples[/yellow]")
    
    # Setup output directory
    output_dir = output_config.get('dir', './output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments
    console.print("\n[blue]Configuring training...[/blue]")
    
    # Check BF16 support (Blackwell supports it)
    use_bf16 = is_bfloat16_supported()
    console.print(f"  Using BF16: {use_bf16}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_config.get('epochs', 3),
        per_device_train_batch_size=train_config.get('batch_size', 4),
        gradient_accumulation_steps=train_config.get('gradient_accumulation', 4),
        learning_rate=float(train_config.get('learning_rate', 2e-4)),
        warmup_ratio=train_config.get('warmup_ratio', 0.03),
        logging_steps=10,
        save_steps=output_config.get('save_steps', 500),
        save_total_limit=3,
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="adamw_8bit",
        weight_decay=train_config.get('weight_decay', 0.01),
        lr_scheduler_type="cosine",
        seed=42,
        report_to=config.get('logging', {}).get('report_to', ['tensorboard']),
        max_grad_norm=train_config.get('max_grad_norm', 1.0),
    )
    
    # Determine text field
    text_field = data_config.get('text_field', None)
    if text_field is None:
        if 'text' in dataset.column_names:
            text_field = 'text'
        else:
            text_field = None
    
    # Initialize trainer
    console.print("\n[blue]Initializing trainer...[/blue]")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=model_config.get('max_seq_length', 16384),
        dataset_text_field=text_field,
    )
    
    console.print("[green]✓ Trainer initialized[/green]")
    
    # Start training
    console.print("\n" + "="*70)
    console.print("[bold cyan]Starting Training[/bold cyan]")
    console.print("="*70 + "\n")
    
    start_time = datetime.now()
    
    try:
        if args.resume:
            console.print(f"Resuming from: {args.resume}")
            trainer.train(resume_from_checkpoint=args.resume)
        else:
            trainer.train()
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        console.print("Saving current checkpoint...")
        trainer.save_model(os.path.join(output_dir, "checkpoint-interrupted"))
        return 1
    
    elapsed = datetime.now() - start_time
    
    console.print("\n" + "="*70)
    console.print("[bold green]Training Complete![/bold green]")
    console.print("="*70)
    console.print(f"\nTotal training time: {elapsed}")
    
    # Save final model
    final_path = os.path.join(output_dir, "checkpoint-final")
    console.print(f"\n[blue]Saving final model to {final_path}...[/blue]")
    
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    console.print(f"[green]✓ Model saved to: {final_path}[/green]")
    
    # Save training summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "base_model": model_config['base_model'],
        "lora_rank": lora_config['r'],
        "lora_alpha": lora_config['alpha'],
        "epochs": train_config['epochs'],
        "training_time_seconds": elapsed.total_seconds(),
        "dataset_size": len(dataset),
        "output_path": final_path,
        "gpu": gpu_name,
    }
    
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    console.print(f"[green]✓ Training summary saved to: {summary_path}[/green]")
    
    console.print("\n[bold]Next steps:[/bold]")
    console.print(f"  1. Test the model: vllm serve {final_path} --port 5000")
    console.print(f"  2. Merge adapters: python merge-adapter.py --adapter {final_path} --output ./merged-model")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
