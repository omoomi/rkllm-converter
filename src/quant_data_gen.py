"""
Quantization data generator for RKLLM converter.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from rich.console import Console
from tqdm import tqdm

from .config import DEFAULT_CALIBRATION_SAMPLES, ConversionConfig


def generate_quantization_data(
    model_path: str,
    output_path: str,
    num_samples: int = 20,
    max_new_tokens: int = 128,
    temperature: float = 0.6,
    repetition_penalty: float = 1.1,
    apply_chat_template: bool = True,
    device: str = "cuda",
    token: Optional[str] = None,
    samples: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generate quantization calibration data.
    
    Args:
        model_path: Path to the model (local or HF ID)
        output_path: Path to save data_quant.json
        num_samples: Number of calibration samples to generate
        max_new_tokens: Maximum tokens to generate for each sample
        temperature: Temperature for generation
        repetition_penalty: Repetition penalty
        apply_chat_template: Whether to apply chat template
        device: Device to run on (cuda/cpu)
        token: HF token for gated models
        samples: Custom samples to use (default uses DEFAULT_CALIBRATION_SAMPLES)
    
    Returns:
        Path to generated data_quant.json
    """
    console = Console()
    
    if samples is None:
        samples = DEFAULT_CALIBRATION_SAMPLES[:num_samples]
    
    num_samples = min(num_samples, len(samples))
    
    console.print(f"[bold cyan]Loading model from:[/bold cyan] {model_path}")
    console.print(f"[bold cyan]Device:[/bold cyan] {device}")
    
    if not torch.cuda.is_available() and device == "cuda":
        console.print("[yellow]CUDA not available, falling back to CPU[/yellow]")
        device = "cpu"
    
    try:
        with console.status("[bold green]Loading tokenizer..."):
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                token=token
            )
        
        with console.status("[bold green]Loading model..."):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                token=token
            )
            model = model.to(device)
            model = model.eval()
    except Exception as e:
        console.print(f"[bold red]Error loading model:[/bold red] {e}")
        raise
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "top_k": 1,
        "temperature": temperature,
        "do_sample": True,
        "repetition_penalty": repetition_penalty
    }
    
    calidata = []
    
    console.print(f"\n[bold cyan]Generating {num_samples} calibration samples...[/bold cyan]\n")
    
    for idx, sample in enumerate(tqdm(samples[:num_samples], desc="Generating samples")):
        question = sample.get("input", "").strip()
        
        if apply_chat_template:
            messages = [{"role": "user", "content": question}]
            messages = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        else:
            messages = question
        
        try:
            inputs = tokenizer(messages, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            target = result[len(messages):] if len(result) > len(messages) else ""
            calidata.append({"input": messages, "target": target})
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to generate sample {idx}: {e}[/yellow]")
            calidata.append({"input": messages, "target": ""})
    
    output_file = Path(output_path).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(calidata, f, ensure_ascii=False, indent=2)
    
    console.print(f"\n[bold green]✓ Calibration data saved to:[/bold green] {output_file.absolute()}")
    console.print(f"[bold cyan]  Total samples:[/bold cyan] {len(calidata)}")
    
    return str(output_file.absolute())


def load_custom_dataset(dataset_path: str) -> List[Dict[str, str]]:
    """
    Load custom calibration dataset from JSON file.
    
    Expected format:
    [
        {"input": "User prompt...", "target": "Expected response..."},
        ...
    ]
    
    Args:
        dataset_path: Path to JSON file
    
    Returns:
        List of calibration samples
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of objects")
    
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each dataset item must be a dictionary")
        if "input" not in item:
            raise ValueError("Each dataset item must have an 'input' field")
    
    return data


def generate_quantization_data_from_config(
    config: ConversionConfig,
    output_dir: str = "."
) -> str:
    """
    Generate quantization data using ConversionConfig.
    
    Args:
        config: ConversionConfig object
        output_dir: Directory to save data_quant.json
    
    Returns:
        Path to generated data_quant.json
    """
    output_path = os.path.join(output_dir, "data_quant.json")
    
    if config.dataset_path:
        samples = load_custom_dataset(config.dataset_path)
    else:
        samples = DEFAULT_CALIBRATION_SAMPLES
    
    return generate_quantization_data(
        model_path=config.model_path,
        output_path=output_path,
        num_samples=20,
        apply_chat_template=True,
        device=config.device,
        token=os.environ.get("HF_TOKEN"),
        samples=samples
    )
