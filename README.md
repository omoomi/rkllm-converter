# RKLLM Model Converter

Convert HuggingFace models to `.rkllm` format for Rockchip NPU deployment on RK3588, RK3576, RK3562, and RV1126B platforms.

## Features

- **Auto-download models** from HuggingFace Hub to default cache directory
- **Multiple quantization formats**: w8a8 (better quality) and w4a16 (smaller size)
- **LoRA adapter support** for fine-tuned models
- **Context length customization** (32 to 16384 tokens, included in output filename)
- **CPU and GPU support** for model loading and quantization data generation
- **Dry-run mode** to preview conversion settings
- **Rich CLI interface** with progress bars and colored output
- **HuggingFace token support** for gated models
- **Auto-generated output filenames** using model name (e.g., `Qwen3-1.7B_w8a8_ctx4096_RK3588.rkllm`)

## Requirements

- Python 3.9, 3.10, 3.11, or 3.12
- CUDA-capable GPU (optional, for faster conversion)
- Sufficient RAM (8GB+ recommended for 1-3B models)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd rkllm-converter
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install PyTorch

For CPU-only installation:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For CUDA 12.4 (recommended if you have an NVIDIA GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> **Note:** Check [PyTorch's installation guide](https://pytorch.org/get-started/locally/) for other CUDA versions.

### 4. Install rkllm_toolkit

The `rkllm_toolkit` package is not available on PyPI. You need to download it from the [RKLLM GitHub repository](https://github.com/airockchip/rknn-llm):

1. Visit: https://github.com/airockchip/rknn-llm/tree/main/rkllm-toolkit/packages
2. Download the appropriate `.whl` file for your Python version:
   - `rkllm_toolkit-1.2.3-cp39-cp39-linux_x86_64.whl`
   - `rkllm_toolkit-1.2.3-cp310-cp310-linux_x86_64.whl`
   - `rkllm_toolkit-1.2.3-cp311-cp311-linux_x86_64.whl`
   - `rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl`
3. Install it:
   ```bash
   pip install path/to/rkllm_toolkit-*.whl --no-deps
   ```

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
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

**Output filename:** `Qwen3-1.7B_w8a8_ctx4096_RK3588.rkllm`

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

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model` | `-m` | HuggingFace model name or local path | **(required)** |
| `--lora` | `-l` | LoRA adapter path | None |
| `--quantization` | `-q` | Quantization format (`w8a8`, `w4a16`) | `w8a8` |
| `--context-length` | `-c` | Maximum context length (32-16384) | `4096` |
| `--platform` | `-p` | Target platform (`RK3588`, `RK3576`, `RK3562`, `RV1126B`) | `RK3588` |
| `--npu-cores` | `-n` | Number of NPU cores | `3` |
| `--output` | `-o` | Output path for .rkllm file | Auto-generated |
| `--dataset` | | Custom calibration dataset path (JSON) | Built-in |
| `--quantization-alg` | | Algorithm (`normal`, `grq`) | Auto-selected |
| `--dry-run` | | Preview settings without executing | False |
| `--device` | | Device (`cpu`, `cuda`) | `cuda` |
| `--dtype` | | Data type (`float32`, `float16`, `bfloat16`) | `float32` |
| `--skip-data-gen` | | Skip quantization data generation | False |

## Supported Models

The converter supports all models compatible with RKLLM toolkit:

- LLaMA models (LLaMA 2, LLaMA 3, etc.)
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

## Context Length

The RKLLM toolkit supports context lengths from **32 to 16384 tokens**.

| Use Case | Recommended Context Length |
|----------|---------------------------|
| Simple Q&A | 512 - 2048 |
| Chatbots | 2048 - 4096 |
| Document Analysis | 4096 - 8192 |
| Long-form Content | 8192 - 16384 |

## Output Filename Format

When no custom output path is specified, filenames are auto-generated as:

```
{model_name}_{quantization}_ctx{context_length}_{platform}.rkllm
```

Examples:
- `Qwen3-1.7B_w8a8_ctx4096_RK3588.rkllm`
- `Llama-3-8B_w4a16_ctx8192_RK3576.rkllm`
- `my-custom-model_w8a8_ctx2048_RK3588.rkllm`

## HuggingFace Token

For gated models (like LLaMA), set the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN=your_token_here
python -m src --model meta-llama/Llama-2-7b-hf
```

## Model Cache

Models are automatically downloaded to HuggingFace's default cache directory:

- **Linux:** `~/.cache/huggingface/hub/`
- **macOS:** `~/Library/Caches/huggingface/hub/`
- **Windows:** `C:\Users\<username>\.cache\huggingface\hub\`

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

## Troubleshooting

### Context Length Error

If you see `ERROR: max_context must be within [32, 16384]`, the requested context length exceeds the maximum supported by RKLLM. Use a smaller value:

```bash
# Instead of --context-length 65536
python -m src --model Qwen/Qwen3-1.7B --context-length 16384
```

### Quantization Data Generation Timeout on CPU

Generating calibration data on CPU can be very slow for large models (10+ minutes for 1-3B models). Solutions:
- Use GPU if available: Remove `--device cpu` to use default CUDA mode
- Skip data generation: Use `--skip-data-gen` if `data_quant.json` already exists
- Reduce sample count: Modify `quant_data_gen.py` to use fewer calibration samples

### CUDA Compatibility Issues

If you encounter CUDA errors with newer GPUs (e.g., RTX 50-series):
- The current stable PyTorch may not support your GPU's compute capability
- Use `--device cpu` to run conversion on CPU (slower but reliable)
- Or install PyTorch nightly build: `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124`

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
pip install path/to/rkllm_toolkit-*.whl --no-deps
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
