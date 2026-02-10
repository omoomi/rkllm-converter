# RKLLM Model Converter

Convert HuggingFace models to `.rkllm` format for Rockchip NPU deployment on RK3588, RK3576, RK3562, and RV1126B platforms.

## Features

- **Auto-download models** from HuggingFace Hub to default cache directory
- **Multiple quantization formats**: w8a8 (better quality) and w4a16 (smaller size)
- **LoRA adapter support** for fine-tuned models
- **Context length customization** for various use cases (included in output filename)
- **CPU and GPU support** for model loading and quantization data generation
- **Dry-run mode** to preview conversion settings
- **Rich CLI interface** with progress bars and colored output
- **HuggingFace token support** for gated models

## Requirements

- Python 3.9, 3.10, 3.11, or 3.12
- CUDA-capable GPU (optional, for faster conversion)
- Sufficient RAM (8GB+ recommended for 1-3B models)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd convert-to-rkllm
```

### 2. Run the setup script

```bash
./scripts/setup_env.sh
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install rkllm_toolkit

The `rkllm_toolkit` package is not available on PyPI. You need to download it from the [RKLLM GitHub repository](https://github.com/airockchip/rknn-llm):

1. Visit: https://github.com/airockchip/rknn-llm/tree/main/rkllm-toolkit/packages
2. Download the appropriate `.whl` file for your Python version:
   - `rkllm_toolkit-1.2.3-cp39-cp39-linux_x86_64.whl`
   - `rkllm_toolkit-1.2.3-cp310-cp310-linux_x86_64.whl`
   - `rkllm_toolkit-1.2.3-cp311-cp311-linux_x86_64.whl`
   - `rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl`
3. Install it:
   ```bash
   pip install path/to/rkllm_toolkit-*.whl
   ```

## Usage

### Basic Conversion

Convert a model with default settings:

```bash
python -m src \
  --model Qwen/Qwen3-1.7B \
  --quantization w8a8 \
  --context-length 4096
```

### With LoRA Adapter

Convert a model with a LoRA adapter:

```bash
python -m src \
  --model Qwen/Qwen3-1.7B \
  --lora organization/my-lora-adapter \
  --quantization w8a8
```

### Custom Output Path

Specify a custom output path:

```bash
python -m src \
  --model Qwen/Qwen3-1.7B \
  --output /path/to/output.rkllm
```

### Dry Run

Preview conversion settings without executing:

```bash
python -m src \
  --model Qwen/Qwen3-1.7B \
  --quantization w8a8 \
  --dry-run
```

### Using CPU

Convert using CPU (slower, no GPU required):

```bash
python -m src \
  --model Qwen/Qwen3-1.7B \
  --device cpu
```

### Custom Calibration Dataset

Use a custom calibration dataset:

```bash
python -m src \
  --model Qwen/Qwen3-1.7B \
  --dataset /path/to/custom_data.json
```

### All Options

```bash
python -m src --help
```

**Options:**

- `--model`, `-m`: HuggingFace model name or local path (required)
- `--lora`, `-l`: LoRA adapter path (optional)
- `--quantization`, `-q`: Quantization format `w8a8|w4a16` (default: w8a8)
- `--context-length`, `-c`: Maximum context length (default: 4096)
- `--platform`, `-p`: Target platform `RK3588|RK3576|RK3562|RV1126B` (default: RK3588)
- `--npu-cores`, `-n`: Number of NPU cores (default: 3)
- `--output`, `-o`: Output path for .rkllm file
- `--dataset`: Custom calibration dataset path (JSON format)
- `--quantization-alg`: `normal|grq` (default: auto-selected)
- `--dry-run`: Preview conversion settings without executing
- `--device`: `cpu|cuda` (default: cuda)
- `--dtype`: `float32|float16|bfloat16` (default: float32)
- `--skip-data-gen`: Skip quantization data generation

## Supported Models

The converter supports all models compatible with RKLLM toolkit:

- LLaMA models
- Qwen2/Qwen2.5/Qwen3
- Phi2/Phi3
- ChatGLM3-6B
- Gemma2/Gemma3
- InternLM2 models
- MiniCPM3/MiniCPM4
- And more...

See the [RKLLM documentation](https://github.com/airockchip/rknn-llm) for the full list.

## Quantization Formats

### w8a8 (Recommended)
- 8-bit weights and activations
- Better quality, larger file size
- Recommended for most use cases
- Uses `normal` quantization algorithm

### w4a16
- 4-bit weights, 16-bit activations
- Smaller file size, acceptable quality
- Uses `grq` quantization algorithm
- Good for memory-constrained deployments

## HuggingFace Token

For gated models (like LLaMA), set the `HF_TOKEN` environment variable:

```bash
 export HF_TOKEN=your_token_here
 python -m src --model meta-llama/Llama-2-7b-hf
```

## Model Cache

Models are automatically downloaded to HuggingFace's default cache directory:

- Linux: `~/.cache/huggingface/hub/`
- macOS: `~/Library/Caches/huggingface/hub/`
- Windows: `C:\Users\<username>\.cache\huggingface\hub\`

## Testing

Run unit tests:

```bash
pip install -r dev-requirements.txt
pytest tests/test_converter.py -v
```

Run integration tests:

```bash
pytest tests/test_integration.py -v
```

## Example: Convert Qwen3-1.7B

```bash
# Activate virtual environment
 source venv/bin/activate

# Convert with w8a8 quantization
python -m src \
  --model Qwen/Qwen3-1.7B \
  --quantization w8a8 \
  --context-length 4096 \
  --platform RK3588 \
  --npu-cores 3
```

This will generate `Qwen3-1.7B_w8a8_ctx4096_RK3588.rkllm` in the current directory.

### Large Context Length

Convert with extended context length (e.g., 65536 tokens):

```bash
# Activate virtual environment
source venv/bin/activate

# Convert with 65536 context length
python -m src \
  --model Qwen/Qwen3-1.7B \
  --quantization w8a8 \
  --context-length 65536 \
  --platform RK3588
```

This will generate `Qwen3-1.7B_w8a8_ctx65536_RK3588.rkllm` in the current directory.

## Troubleshooting

### Quantization Data Generation Timeout on CPU

Generating calibration data on CPU can be very slow for large models (10+ minutes for 1-3B models). Solutions:
- Use GPU if available: Remove `--device cpu` to use default CUDA mode
- Skip data generation: Use `--skip-data-gen` if `data_quant.json` already exists
- Reduce sample count: Modify `quant_data_gen.py` to use fewer calibration samples

### CUDA Out of Memory

If you encounter CUDA OOM errors, try:
- Using CPU mode: `--device cpu`
- Reducing context length: `--context-length 2048`
- Closing other GPU-intensive applications

### Model Download Fails

- Check your internet connection
- Verify the model name is correct
- Set `HF_TOKEN` for gated models

### Import Error: No module named 'rkllm'

Make sure you have installed the `rkllm_toolkit` wheel file:

```bash
pip install path/to/rkllm_toolkit-*.whl
```

## Development

To install development dependencies:

```bash
pip install -r dev-requirements.txt
```

To format code:

```bash
black src/
```

To lint code:

```bash
flake8 src/
```

## License

MIT License

## Contributing

Contributions are welcome! Please submit pull requests to the repository.

## Acknowledgments

- [RKLLM Toolkit](https://github.com/airockchip/rknn-llm) by Rockchip
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Rich](https://github.com/Textualize/rich) for beautiful CLI output
