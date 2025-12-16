#!/usr/bin/env python3
"""
merge-adapter.py - Merge LoRA adapters into base model

Creates a standalone fine-tuned model by merging trained LoRA adapters
into the base Fara-7B model.

Usage:
    python merge-adapter.py --adapter ./output/checkpoint-final --output ./merged-model
    python merge-adapter.py --adapter ./output/checkpoint-500 --output ./merged-500 --base-model microsoft/Fara-7B

Author: Magentic-UI DGX Stack
Target: NVIDIA DGX Spark
"""

import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def print_banner():
    """Print startup banner."""
    console.print("\n[bold cyan]LoRA Adapter Merger for Fara-7B[/bold cyan]")
    console.print("="*50 + "\n")


def check_gpu():
    """Check GPU availability."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[green]✓ GPU:[/green] {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    else:
        console.print("[yellow]⚠ No GPU detected. Merging on CPU (slower).[/yellow]")
        return False


def merge_adapter(
    adapter_path: str,
    output_path: str,
    base_model: str = "microsoft/Fara-7B",
    max_seq_length: int = 16384,
    save_fp16: bool = True
):
    """Merge LoRA adapter into base model."""
    
    console.print(f"\n[blue]Configuration:[/blue]")
    console.print(f"  Adapter:     {adapter_path}")
    console.print(f"  Base model:  {base_model}")
    console.print(f"  Output:      {output_path}")
    console.print(f"  Precision:   {'FP16' if save_fp16 else 'FP32'}")
    
    # Import Unsloth
    console.print("\n[blue]Loading Unsloth...[/blue]")
    try:
        from unsloth import FastLanguageModel
        console.print("[green]✓ Unsloth loaded[/green]")
    except ImportError:
        console.print("[red]Error: Unsloth not installed[/red]")
        console.print("Install with: pip install -r requirements.txt")
        return False
    
    # Load model with adapter
    console.print("\n[blue]Loading model with adapter...[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading...", total=None)
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=max_seq_length,
            dtype=torch.float16 if save_fp16 else None,
            load_in_4bit=False,
        )
        
        progress.update(task, completed=True)
    
    console.print("[green]✓ Model loaded with adapter[/green]")
    
    # Merge adapter into model
    console.print("\n[blue]Merging adapter...[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Merging...", total=None)
        
        # Unsloth's merge_and_unload
        model = model.merge_and_unload()
        
        progress.update(task, completed=True)
    
    console.print("[green]✓ Adapter merged successfully[/green]")
    
    # Save merged model
    console.print(f"\n[blue]Saving merged model to {output_path}...[/blue]")
    
    os.makedirs(output_path, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Saving...", total=None)
        
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        progress.update(task, completed=True)
    
    console.print(f"[green]✓ Merged model saved to: {output_path}[/green]")
    
    # Save merge metadata
    metadata = {
        "merged_at": datetime.now().isoformat(),
        "adapter_path": adapter_path,
        "base_model": base_model,
        "max_seq_length": max_seq_length,
        "precision": "fp16" if save_fp16 else "fp32",
    }
    
    metadata_path = os.path.join(output_path, "merge_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print model size
    model_size = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if f.endswith('.safetensors') or f.endswith('.bin')
    ) / 1e9
    
    console.print(f"\n[green]Model size: {model_size:.2f} GB[/green]")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Merge LoRA adapters into base Fara-7B model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic merge
  python merge-adapter.py --adapter ./output/checkpoint-final --output ./merged-model
  
  # Specify base model
  python merge-adapter.py --adapter ./adapter --output ./merged --base-model microsoft/Fara-7B
  
  # Keep FP32 precision
  python merge-adapter.py --adapter ./adapter --output ./merged --no-fp16
        """
    )
    
    parser.add_argument(
        '--adapter', '-a',
        type=str,
        required=True,
        help='Path to trained LoRA adapter checkpoint'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output path for merged model'
    )
    parser.add_argument(
        '--base-model', '-b',
        type=str,
        default='microsoft/Fara-7B',
        help='Base model name (default: microsoft/Fara-7B)'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=16384,
        help='Maximum sequence length (default: 16384)'
    )
    parser.add_argument(
        '--no-fp16',
        action='store_true',
        help='Save in FP32 instead of FP16'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Validate adapter path
    if not os.path.exists(args.adapter):
        console.print(f"[red]Error: Adapter path does not exist: {args.adapter}[/red]")
        return 1
    
    # Check for adapter config
    adapter_config = os.path.join(args.adapter, 'adapter_config.json')
    if not os.path.exists(adapter_config):
        console.print(f"[yellow]Warning: No adapter_config.json found. This may not be a LoRA adapter.[/yellow]")
    
    # Check GPU
    check_gpu()
    
    # Perform merge
    success = merge_adapter(
        adapter_path=args.adapter,
        output_path=args.output,
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        save_fp16=not args.no_fp16
    )
    
    if success:
        console.print("\n[bold green]Merge complete![/bold green]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  Test with vLLM: vllm serve {args.output} --port 5000")
        console.print(f"  Or copy to models directory for use with the stack")
        return 0
    else:
        console.print("\n[bold red]Merge failed![/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
