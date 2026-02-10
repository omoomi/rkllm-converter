"""
Unit tests for RKLLM converter components.
"""

import pytest
import json
from pathlib import Path
import tempfile
import os

from src.config import ConversionConfig, DEFAULT_CALIBRATION_SAMPLES
from src.utils import (
    is_huggingface_model_path,
    format_file_size,
    get_hf_token
)
from src.quant_data_gen import load_custom_dataset


class TestConversionConfig:
    """Test ConversionConfig class."""
    
    def test_config_initialization(self):
        config = ConversionConfig(
            model_path="Qwen/Qwen3-1.7B",
            quantization="w8a8",
            context_length=4096
        )
        
        assert config.model_path == "Qwen/Qwen3-1.7B"
        assert config.quantization == "w8a8"
        assert config.context_length == 4096
        assert config.platform == "RK3588"
        assert config.npu_cores == 3
    
    def test_get_quantization_algorithm(self):
        config_w8a8 = ConversionConfig(
            model_path="Qwen/Qwen3-1.7B",
            quantization="w8a8"
        )
        assert config_w8a8.get_quantization_algorithm() == "normal"
        
        config_w4a16 = ConversionConfig(
            model_path="Qwen/Qwen3-1.7B",
            quantization="w4a16"
        )
        assert config_w4a16.get_quantization_algorithm() == "grq"
        
        config_custom = ConversionConfig(
            model_path="Qwen/Qwen3-1.7B",
            quantization="w8a8",
            quantization_algorithm="grq"
        )
        assert config_custom.get_quantization_algorithm() == "grq"
    
    def test_get_output_filename(self):
        config = ConversionConfig(model_path="Qwen/Qwen3-1.7B", quantization="w8a8", context_length=4096)
        filename = config.get_output_filename()
        assert "Qwen3-1.7B" in filename
        assert "w8a8" in filename
        assert "ctx4096" in filename
        assert "RK3588" in filename
        assert filename.endswith(".rkllm")
    
    def test_get_output_filename_with_lora(self):
        config = ConversionConfig(
            model_path="Qwen/Qwen3-1.7B",
            lora_path="my/lora/adapter",
            quantization="w8a8"
        )
        filename = config.get_output_filename()
        assert "adapter" in filename
        assert "w8a8" in filename
    
    def test_get_output_filename_large_context(self):
        config = ConversionConfig(
            model_path="Qwen/Qwen3-1.7B",
            quantization="w8a8",
            context_length=65536
        )
        filename = config.get_output_filename()
        assert "Qwen3-1.7B" in filename
        assert "w8a8" in filename
        assert "ctx65536" in filename
        assert "RK3588" in filename
        assert filename.endswith(".rkllm")


class TestUtils:
    """Test utility functions."""
    
    def test_is_huggingface_model_path(self):
        assert is_huggingface_model_path("Qwen/Qwen3-1.7B") == True
        assert is_huggingface_model_path("org/model-name") == True
        assert is_huggingface_model_path("/local/path/to/model") == False
        assert is_huggingface_model_path("./model") == False
    
    def test_format_file_size(self):
        assert format_file_size(500) == "500.00 B"
        assert format_file_size(1024) == "1.00 KB"
        assert format_file_size(1536) == "1.50 KB"
        assert format_file_size(1024 * 1024) == "1.00 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.00 GB"
    
    def test_get_hf_token(self):
        token = get_hf_token()
        if "HF_TOKEN" in os.environ:
            assert token == os.environ["HF_TOKEN"]
        else:
            assert token is None


class TestQuantizationDataGenerator:
    """Test quantization data generator."""
    
    def test_default_calibration_samples(self):
        assert len(DEFAULT_CALIBRATION_SAMPLES) > 0
        assert all("input" in sample for sample in DEFAULT_CALIBRATION_SAMPLES)
    
    def test_load_custom_dataset(self, tmp_path):
        custom_data = [
            {"input": "Hello", "target": "Hi there"},
            {"input": "Test", "target": "Response"}
        ]
        
        dataset_file = tmp_path / "custom_dataset.json"
        with open(dataset_file, "w") as f:
            json.dump(custom_data, f)
        
        loaded = load_custom_dataset(str(dataset_file))
        assert len(loaded) == 2
        assert loaded[0]["input"] == "Hello"
        assert loaded[1]["target"] == "Response"
    
    def test_load_custom_dataset_invalid_format(self, tmp_path):
        invalid_data = {"not": "a list"}
        
        dataset_file = tmp_path / "invalid_dataset.json"
        with open(dataset_file, "w") as f:
            json.dump(invalid_data, f)
        
        with pytest.raises(ValueError, match="must be a list"):
            load_custom_dataset(str(dataset_file))
    
    def test_load_custom_dataset_missing_input(self, tmp_path):
        invalid_data = [{"output": "no input field"}]
        
        dataset_file = tmp_path / "missing_input.json"
        with open(dataset_file, "w") as f:
            json.dump(invalid_data, f)
        
        with pytest.raises(ValueError, match="input"):
            load_custom_dataset(str(dataset_file))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
