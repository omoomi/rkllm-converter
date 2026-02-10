"""
Model downloader for HuggingFace models.
"""

import os
from typing import Optional
from huggingface_hub import snapshot_download, hf_hub_download
from rich.console import Console


def download_model(
    model_id: str,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> str:
    """
    Download a HuggingFace model to the local cache.
    
    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-1.7B")
        token: HuggingFace authentication token (for gated models)
        cache_dir: Custom cache directory (default: ~/.cache/huggingface)
    
    Returns:
        Local path to downloaded model
    """
    console = Console()
    
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    console.print(f"\n[bold cyan]Downloading model:[/bold cyan] {model_id}")
    console.print(f"[bold cyan]Cache directory:[/bold cyan] {cache_dir}\n")
    
    local_path = snapshot_download(
        repo_id=model_id,
        token=token,
        local_files_only=False,
        cache_dir=cache_dir,
        resume_download=True
    )
    
    console.print(f"[bold green]✓ Model downloaded to:[/bold green] {local_path}\n")
    
    return local_path


def download_lora(
    lora_id: str,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> str:
    """
    Download a HuggingFace LoRA adapter to the local cache.
    
    Args:
        lora_id: HuggingFace model ID for LoRA adapter
        token: HuggingFace authentication token
        cache_dir: Custom cache directory (default: ~/.cache/huggingface)
    
    Returns:
        Local path to downloaded LoRA adapter
    """
    console = Console()
    
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    console.print(f"\n[bold cyan]Downloading LoRA adapter:[/bold cyan] {lora_id}")
    console.print(f"[bold cyan]Cache directory:[/bold cyan] {cache_dir}\n")
    
    local_path = snapshot_download(
        repo_id=lora_id,
        token=token,
        local_files_only=False,
        cache_dir=cache_dir,
        resume_download=True
    )
    
    console.print(f"[bold green]✓ LoRA adapter downloaded to:[/bold green] {local_path}\n")
    
    return local_path


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment variables."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def check_model_exists(model_id: str, token: Optional[str] = None) -> bool:
    """
    Check if a model exists on HuggingFace Hub.
    
    Args:
        model_id: HuggingFace model ID
        token: HuggingFace authentication token
    
    Returns:
        True if model exists, False otherwise
    """
    from huggingface_hub import repo_exists
    
    return repo_exists(model_id, token=token)
