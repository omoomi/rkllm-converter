"""
Model converter for RKLLM.
"""

import os
from pathlib import Path
from typing import Optional
from rkllm.api import RKLLM
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from .config import ConversionConfig
from .utils import print_conversion_summary


def convert_model(config: ConversionConfig, quant_data_path: Optional[str]) -> str:
    """
    Convert a HuggingFace model to RKLLM format.

    Args:
        config: ConversionConfig with all settings
        quant_data_path: Path to calibration data JSON file (None if no quantization)

    Returns:
        Path to generated .rkllm file
    """
    console = Console()
    
    console.print("\n[bold cyan]Starting RKLLM conversion...[/bold cyan]\n")
    
    if config.dry_run:
        print_conversion_summary(config)
        console.print("[yellow]Dry-run mode: Skipping actual conversion[/yellow]\n")
        return config.get_output_filename()
    
    llm = RKLLM()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            load_task = progress.add_task("[bold green]Loading model...", total=None)
            
            device = "cuda" if config.device == "cuda" else "cpu"
            
            ret = llm.load_huggingface(
                model=config.model_path,
                model_lora=config.lora_path,
                device=device,
                dtype=config.dtype,
                custom_config=None,
                load_weight=True
            )
            
            progress.update(load_task, completed=100)
            
            if ret != 0:
                console.print("[bold red]✗ Load model failed![/bold red]")
                raise RuntimeError("Failed to load model")
            
            console.print("[bold green]✓ Model loaded successfully[/bold green]")
            
            build_task = progress.add_task("[bold green]Building model...", total=100)
            progress.update(build_task, advance=10)
            
            # Only quantize if quantization is enabled
            do_quant = config.quantization is not None
            qparams = None
            quant_dtype = config.quantization_dtype() if do_quant else None
            quant_alg = config.get_quantization_algorithm() if do_quant else None

            ret = llm.build(
                do_quantization=do_quant,
                optimization_level=config.get_optimization_level(),
                quantized_dtype=quant_dtype,
                quantized_algorithm=quant_alg,
                target_platform=config.platform,
                num_npu_core=config.npu_cores,
                extra_qparams=qparams,
                dataset=quant_data_path if do_quant else None,
                hybrid_rate=0,
                max_context=config.context_length
            )
            
            progress.update(build_task, completed=100)
            
            if ret != 0:
                console.print("[bold red]✗ Build model failed![/bold red]")
                raise RuntimeError("Failed to build model")
            
            console.print("[bold green]✓ Model built successfully[/bold green]")
            
            output_filename = config.get_output_filename()
            if config.output_path:
                output_path = config.output_path
            else:
                output_path = output_filename
            
            export_task = progress.add_task(f"[bold green]Exporting to {output_filename}...", total=None)
            
            ret = llm.export_rkllm(output_path)
            
            if ret != 0:
                console.print("[bold red]✗ Export model failed![/bold red]")
                raise RuntimeError("Failed to export model")
            
            progress.update(export_task, completed=100)
            console.print(f"[bold green]✓ Model exported to:[/bold green] {output_path}")
    
    except Exception as e:
        console.print(f"\n[bold red]Error during conversion:[/bold red] {e}")
        raise
    
    console.print("\n[bold green]Conversion completed successfully![/bold green]\n")
    
    output_file = Path(output_path)
    if output_file.exists():
        size = output_file.stat().st_size
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                console.print(f"[bold cyan]Output file size:[/bold cyan] {size:.2f} {unit}")
                break
            size /= 1024.0
    
    return output_path


ConversionConfig.quantization_dtype = lambda self: self.quantization.replace("w", "W").replace("a", "A") if self.quantization else None
