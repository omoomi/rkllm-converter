"""
RKLLM Model Converter - Convert HuggingFace models to .rkllm format for Rockchip NPU.
"""

__version__ = "0.1.1"
__author__ = "RKLLM Converter Team"

from .config import ConversionConfig, DEFAULT_CALIBRATION_SAMPLES
from .quant_data_gen import generate_quantization_data, load_custom_dataset
from .utils import is_huggingface_model_path, resolve_model_path, get_hf_token

def convert_model(*args, **kwargs):
    """Wrapper for convert_model that imports rkllm only when needed."""
    from .converter import convert_model as _convert_model
    return _convert_model(*args, **kwargs)

def download_model(*args, **kwargs):
    """Wrapper for download_model."""
    from .downloader import download_model as _download_model
    return _download_model(*args, **kwargs)

def download_lora(*args, **kwargs):
    """Wrapper for download_lora."""
    from .downloader import download_lora as _download_lora
    return _download_lora(*args, **kwargs)

__all__ = [
    "ConversionConfig",
    "DEFAULT_CALIBRATION_SAMPLES",
    "convert_model",
    "download_model",
    "download_lora",
    "generate_quantization_data",
    "load_custom_dataset",
    "is_huggingface_model_path",
    "resolve_model_path",
    "get_hf_token",
]
