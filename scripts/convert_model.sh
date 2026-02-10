#!/bin/bash
# Example conversion script for RKLLM converter

# Activate virtual environment
source venv/bin/activate

# Example 1: Convert Qwen3-1.7B with default settings
echo "Example 1: Convert Qwen3-1.7B (w8a8 quantization, 4096 context length)"
python -m src \
  --model Qwen/Qwen3-1.7B \
  --quantization w8a8 \
  --context-length 4096 \
  --platform RK3588

# Example 2: Convert with custom settings
# echo "Example 2: Convert with w4a16 quantization"
# python -m src \
#   --model Qwen/Qwen3-1.7B \
#   --quantization w4a16 \
#   --context-length 8192 \
#   --platform RK3588 \
#   --output /path/to/output.rkllm

# Example 3: Dry run to preview settings
# echo "Example 3: Dry run to preview settings"
# python -m src \
#   --model Qwen/Qwen3-1.7B \
#   --quantization w8a8 \
#   --dry-run

# Example 4: Convert with LoRA adapter
# echo "Example 4: Convert with LoRA adapter"
# python -m src \
#   --model Qwen/Qwen3-1.7B \
#   --lora organization/my-lora-adapter \
#   --quantization w8a8 \
#   --context-length 4096

# Example 5: Convert using CPU
# echo "Example 5: Convert using CPU (slower, no GPU required)"
# python -m src \
#   --model Qwen/Qwen3-1.7B \
#   --quantization w8a8 \
#   --device cpu

# Uncomment the examples you want to use
