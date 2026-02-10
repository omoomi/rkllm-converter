"""
Utility functions for RKLLM converter.
"""

import os
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download, snapshot_download
from rich.console import Console


def is_huggingface_model_path(path: str) -> bool:
    """Check if the given path is a HuggingFace model ID."""
    path_obj = Path(path).expanduser()
    
    # If it's an absolute path that exists, it's a local path
    if path_obj.is_absolute() and path_obj.exists():
        return False
    
    # If it starts with /, ., or ~, it's likely a local path
    if path.startswith(("/", ".", "~")):
        return False
    
    # If it contains / but doesn't exist locally, it's likely an HF model ID
    # HF model IDs are like "organization/model-name"
    return "/" in path and not path_obj.exists()


def resolve_model_path(model_path: str, token: Optional[str] = None) -> str:
    """
    Resolve model path - download from HF if needed, otherwise return local path.
    
    Uses HuggingFace's default cache directory.
    """
    if is_huggingface_model_path(model_path):
        console = Console()
        with console.status(f"[bold green]Downloading model {model_path}..."):
            cache_dir = snapshot_download(
                repo_id=model_path,
                token=token,
                local_files_only=False,
                cache_dir=os.path.expanduser("~/.cache/huggingface")
            )
        return cache_dir
    else:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        return str(path.absolute())


def resolve_lora_path(lora_path: Optional[str], token: Optional[str] = None) -> Optional[str]:
    """
    Resolve LoRA adapter path - download from HF if needed, otherwise return local path.
    """
    if not lora_path:
        return None
    
    if is_huggingface_model_path(lora_path):
        console = Console()
        with console.status(f"[bold green]Downloading LoRA adapter {lora_path}..."):
            cache_dir = snapshot_download(
                repo_id=lora_path,
                token=token,
                local_files_only=False,
                cache_dir=os.path.expanduser("~/.cache/huggingface")
            )
        return cache_dir
    else:
        path = Path(lora_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"LoRA path not found: {path}")
        return str(path.absolute())


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment variable."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def validate_output_path(output_path: str) -> str:
    """Validate and prepare output path."""
    path = Path(output_path).expanduser()
    
    if path.exists() and path.is_file():
        raise FileExistsError(f"Output file already exists: {output_path}")
    
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    
    return str(path.absolute())


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def print_conversion_summary(config):
    """Print conversion summary for dry-run mode."""
    console = Console()
    console.print("\n[bold cyan]Conversion Summary:[/bold cyan]\n")
    console.print(f"  Model: {config.model_path}")
    if config.lora_path:
        console.print(f"  LoRA Adapter: {config.lora_path}")
    console.print(f"  Quantization: {config.quantization}")
    console.print(f"  Context Length: {config.context_length}")
    console.print(f"  Platform: {config.platform}")
    console.print(f"  NPU Cores: {config.npu_cores}")
    console.print(f"  Device: {config.device}")
    console.print(f"  Data Type: {config.dtype}")
    console.print(f"  Quantization Algorithm: {config.get_quantization_algorithm()}")
    console.print(f"  Optimization Level: {config.get_optimization_level()}")
    if config.output_path:
        console.print(f"  Output Path: {config.output_path}")
    else:
        console.print(f"  Output Path: {config.get_output_filename()}")
    console.print("")
