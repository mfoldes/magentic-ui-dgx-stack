#!/usr/bin/env python3
"""
prepare-dataset.py - Prepare datasets for Fara-7B fine-tuning

Converts various data formats to the training format expected by Fara-7B.
Supports trajectory data (CUA format), chat/conversation format, and raw text.

Usage:
    python prepare-dataset.py --input data.json --output training.jsonl --format fara
    python prepare-dataset.py --input conversations.jsonl --output training.jsonl --format chat
    python prepare-dataset.py --input folder/ --output training.jsonl --format auto

Author: Magentic-UI DGX Stack
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


def format_fara_trajectory(task: str, trajectory: List[Dict[str, Any]]) -> str:
    """Format a trajectory into Fara-7B training format."""
    parts = [f"Task: {task}"]
    
    for i, step in enumerate(trajectory):
        thought = step.get('thought', '')
        action = step.get('action', '')
        
        # Include observation description if available (not the raw screenshot)
        observation_desc = step.get('observation_description', '')
        if observation_desc:
            parts.append(f"Observation: {observation_desc}")
        
        parts.append(f"Thought: {thought}")
        parts.append(f"Action: {action}")
    
    return "\n".join(parts)


def format_chat_messages(messages: List[Dict[str, str]]) -> str:
    """Format chat messages into training format."""
    parts = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'system':
            parts.append(f"System: {content}")
        elif role == 'user':
            parts.append(f"User: {content}")
        elif role == 'assistant':
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"{role}: {content}")
    
    return "\n".join(parts)


def process_fara_format(data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Process Fara trajectory format."""
    if 'trajectory' not in data:
        return None
    
    task = data.get('task', 'Complete the task')
    trajectory = data['trajectory']
    
    text = format_fara_trajectory(task, trajectory)
    return {"text": text}


def process_chat_format(data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Process chat/conversation format."""
    messages = data.get('messages', data.get('conversation', []))
    if not messages:
        return None
    
    text = format_chat_messages(messages)
    return {"text": text}


def process_text_format(data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Process raw text format."""
    if 'text' in data:
        return {"text": data['text']}
    elif 'content' in data:
        return {"text": data['content']}
    return None


def detect_format(data: Dict[str, Any]) -> str:
    """Auto-detect data format."""
    if 'trajectory' in data:
        return 'fara'
    elif 'messages' in data or 'conversation' in data:
        return 'chat'
    elif 'text' in data or 'content' in data:
        return 'text'
    else:
        return 'unknown'


def process_file(
    input_path: str, 
    output_path: str, 
    data_format: str = 'auto',
    max_samples: Optional[int] = None
) -> int:
    """Process input file and write training data."""
    
    samples = []
    skipped = 0
    
    console.print(f"\n[blue]Processing:[/blue] {input_path}")
    console.print(f"[blue]Format:[/blue] {data_format}")
    
    # Read input data
    with open(input_path, 'r', encoding='utf-8') as f:
        # Try to detect if it's JSONL or regular JSON
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # Regular JSON array
            data_list = json.load(f)
        else:
            # JSONL format
            data_list = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
    
    console.print(f"[blue]Found:[/blue] {len(data_list)} records")
    
    # Process each record
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(data_list))
        
        for data in data_list:
            # Detect or use specified format
            if data_format == 'auto':
                fmt = detect_format(data)
            else:
                fmt = data_format
            
            # Process based on format
            if fmt == 'fara':
                result = process_fara_format(data)
            elif fmt == 'chat':
                result = process_chat_format(data)
            elif fmt == 'text':
                result = process_text_format(data)
            else:
                result = None
                skipped += 1
            
            if result:
                samples.append(result)
            else:
                skipped += 1
            
            progress.update(task, advance=1)
            
            # Check max samples limit
            if max_samples and len(samples) >= max_samples:
                break
    
    # Write output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    console.print(f"\n[green]✓ Processed:[/green] {len(samples)} samples")
    if skipped > 0:
        console.print(f"[yellow]⚠ Skipped:[/yellow] {skipped} records")
    console.print(f"[green]✓ Output:[/green] {output_path}")
    
    return len(samples)


def process_folder(
    input_folder: str, 
    output_path: str, 
    data_format: str = 'auto',
    max_samples: Optional[int] = None
) -> int:
    """Process all JSON/JSONL files in a folder."""
    
    input_path = Path(input_folder)
    all_samples = []
    
    # Find all JSON files
    json_files = list(input_path.glob('*.json')) + list(input_path.glob('*.jsonl'))
    
    console.print(f"\n[blue]Found {len(json_files)} JSON files in {input_folder}[/blue]")
    
    for json_file in json_files:
        console.print(f"\nProcessing: {json_file.name}")
        
        # Process file to temp list
        temp_output = str(json_file) + '.temp'
        count = process_file(str(json_file), temp_output, data_format, max_samples)
        
        # Read temp and add to all samples
        if os.path.exists(temp_output):
            with open(temp_output, 'r') as f:
                for line in f:
                    all_samples.append(json.loads(line.strip()))
            os.remove(temp_output)
        
        if max_samples and len(all_samples) >= max_samples:
            all_samples = all_samples[:max_samples]
            break
    
    # Write combined output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    console.print(f"\n[green]✓ Total samples:[/green] {len(all_samples)}")
    console.print(f"[green]✓ Output:[/green] {output_path}")
    
    return len(all_samples)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset for Fara-7B fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process Fara trajectory data
  python prepare-dataset.py --input trajectories.json --output training.jsonl --format fara
  
  # Process chat conversations
  python prepare-dataset.py --input chats.jsonl --output training.jsonl --format chat
  
  # Auto-detect format
  python prepare-dataset.py --input data/ --output training.jsonl --format auto
  
  # Limit samples
  python prepare-dataset.py --input data.json --output training.jsonl --max-samples 1000
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input file or folder path'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='auto',
        choices=['auto', 'fara', 'chat', 'text'],
        help='Input data format (default: auto-detect)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process'
    )
    
    args = parser.parse_args()
    
    console.print("\n[bold cyan]Fara-7B Dataset Preparation Tool[/bold cyan]")
    console.print("="*50)
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        console.print(f"[red]Error: Input path does not exist: {args.input}[/red]")
        return 1
    
    if input_path.is_dir():
        count = process_folder(args.input, args.output, args.format, args.max_samples)
    else:
        count = process_file(args.input, args.output, args.format, args.max_samples)
    
    if count == 0:
        console.print("\n[yellow]Warning: No samples were processed[/yellow]")
        return 1
    
    console.print("\n[bold green]Dataset preparation complete![/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
