"""
Integration tests for RKLLM converter.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.config import ConversionConfig
from src.utils import validate_output_path, resolve_model_path


class TestIntegration:
    """Integration tests for the converter."""
    
    def test_resolve_local_model_path(self, tmp_path):
        """Test resolving a local model path."""
        model_dir = tmp_path / "models" / "test_model"
        model_dir.mkdir(parents=True)
        
        config_file = model_dir / "config.json"
        config_file.write_text('{"model_type": "llama"}')
        
        resolved = resolve_model_path(str(model_dir))
        assert resolved.endswith(str(model_dir))
    
    def test_resolve_nonexistent_path(self):
        """Test that nonexistent paths raise error."""
        with pytest.raises(FileNotFoundError):
            resolve_model_path("/nonexistent/path")
    
    def test_validate_output_path(self, tmp_path):
        """Test output path validation."""
        output_path = tmp_path / "output" / "model.rkllm"
        validated = validate_output_path(str(output_path))
        
        assert Path(validated).parent.exists()
    
    def test_validate_output_path_existing_file(self, tmp_path):
        """Test that existing output files are rejected."""
        existing_file = tmp_path / "existing.rkllm"
        existing_file.write_text("test content")
        
        with pytest.raises(FileExistsError):
            validate_output_path(str(existing_file))
    
    def test_config_dry_run(self):
        """Test dry-run mode configuration."""
        config = ConversionConfig(
            model_path="Qwen/Qwen3-1.7B",
            quantization="w8a8",
            dry_run=True
        )
        
        assert config.dry_run == True
        assert config.model_path == "Qwen/Qwen3-1.7B"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
