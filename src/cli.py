"""
Command-line interface for RKLLM converter.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

from .config import ConversionConfig
from .quant_data_gen import generate_quantization_data_from_config
from .utils import (
    is_huggingface_model_path,
    resolve_model_path,
    resolve_lora_path,
    get_hf_token,
    validate_output_path,
    print_conversion_summary
)

console = Console()


@click.command()
@click.option(
    "--model", "-m",
    required=True,
    help="HuggingFace model name or local path (e.g., Qwen/Qwen3-1.7B)"
)
@click.option(
    "--lora", "-l",
    default=None,
    help="LoRA adapter path (HuggingFace ID or local path)"
)
@click.option(
    "--quantization", "-q",
    type=click.Choice(["w8a8", "w4a16"], case_sensitive=False),
    default="w8a8",
    show_default=True,
    help="Quantization format"
)
@click.option(
    "--context-length", "-c",
    type=int,
    default=4096,
    show_default=True,
    help="Maximum context length"
)
@click.option(
    "--platform", "-p",
    type=click.Choice(["RK3588", "RK3576", "RK3562", "RV1126B"], case_sensitive=True),
    default="RK3588",
    show_default=True,
    help="Target platform"
)
@click.option(
    "--npu-cores", "-n",
    type=int,
    default=3,
    show_default=True,
    help="Number of NPU cores"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output path for .rkllm file (default: auto-generated)"
)
@click.option(
    "--dataset",
    type=click.Path(exists=True),
    default=None,
    help="Custom calibration dataset path (JSON format)"
)
@click.option(
    "--quantization-alg",
    type=click.Choice(["normal", "grq"], case_sensitive=False),
    default=None,
    help="Quantization algorithm (default: auto-selected based on quantization type)"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview conversion settings without executing"
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cuda",
    show_default=True,
    help="Device for model loading"
)
@click.option(
    "--dtype",
    type=click.Choice(["float32", "float16", "bfloat16"], case_sensitive=False),
    default="float32",
    show_default=True,
    help="Data type for model loading"
)
@click.option(
    "--skip-data-gen",
    is_flag=True,
    help="Skip quantization data generation (use existing data_quant.json)"
)
def main(
    model: str,
    lora: Optional[str],
    quantization: str,
    context_length: int,
    platform: str,
    npu_cores: int,
    output: Optional[str],
    dataset: Optional[str],
    quantization_alg: Optional[str],
    dry_run: bool,
    device: str,
    dtype: str,
    skip_data_gen: bool
):
    """
    RKLLM Model Converter - Convert HuggingFace models to .rkllm format for Rockchip NPU.

    Example usage:

    \b
    # Convert base model
    python -m src --model Qwen/Qwen3-1.7B --quantization w8a8

    \b
    # Convert with LoRA adapter
    python -m src --model Qwen/Qwen3-1.7B --lora org/lora-adapter

    \b
    # Dry run to preview settings
    python -m src --model Qwen/Qwen3-1.7B --dry-run
    """
    
    console.print(Panel.fit(
        "[bold cyan]RKLLM Model Converter[/bold cyan]\n"
        "[dim]Convert HuggingFace models to .rkllm format for Rockchip NPU[/dim]",
        title="Welcome",
        border_style="cyan"
    ))
    
    try:
        token = get_hf_token()
        
        config = ConversionConfig(
            model_path=model,
            lora_path=lora,
            quantization=quantization,
            context_length=context_length,
            platform=platform,
            npu_cores=npu_cores,
            output_path=output,
            dataset_path=dataset,
            quantization_algorithm=quantization_alg,
            dry_run=dry_run,
            device=device,
            dtype=dtype
        )
        
        if dry_run:
            print_conversion_summary(config)
            return
        
        console.print(f"\n[bold cyan]Model:[/bold cyan] {model}")
        if lora:
            console.print(f"[bold cyan]LoRA Adapter:[/bold cyan] {lora}")
        console.print(f"[bold cyan]Quantization:[/bold cyan] {quantization}")
        console.print(f"[bold cyan]Context Length:[/bold cyan] {context_length}")
        console.print(f"[bold cyan]Platform:[/bold cyan] {platform}")
        
        model_path = resolve_model_path(model, token)
        console.print(f"[bold green]✓ Model resolved to:[/bold green] {model_path}")
        
        if lora:
            lora_path = resolve_lora_path(lora, token)
            console.print(f"[bold green]✓ LoRA resolved to:[/bold green] {lora_path}")
            config.lora_path = lora_path
        else:
            lora_path = None
        
        config.model_path = model_path
        
        output_dir = "."
        if output:
            output_dir = str(Path(output).parent)
        
        if skip_data_gen and os.path.exists(os.path.join(output_dir, "data_quant.json")):
            quant_data_path = os.path.join(output_dir, "data_quant.json")
            console.print(f"[bold yellow]Using existing calibration data:[/bold yellow] {quant_data_path}")
        else:
            quant_data_path = generate_quantization_data_from_config(config, output_dir)
        
        if output:
            validated_output = validate_output_path(output)
            config.output_path = validated_output
        
        # Lazy import of convert_model to handle missing rkllm_toolkit gracefully
        from .converter import convert_model
        output_file = convert_model(config, quant_data_path)
        
        console.print(Panel.fit(
            f"[bold green]Conversion completed successfully![/bold green]\n\n"
            f"[cyan]Output file:[/cyan] {output_file}",
            title="Success",
            border_style="green"
        ))
        
    except FileNotFoundError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except FileExistsError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        console.print("[yellow]Use --output with a different path or remove the existing file.[/yellow]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Conversion interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")
        import traceback
        console.print(f"\n[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
