#!/bin/bash
# Setup script for RKLLM converter environment

set -e

echo "========================================"
echo "RKLLM Converter Environment Setup"
echo "========================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $PYTHON_VERSION"

# Check if version is supported
if ! echo "$PYTHON_VERSION" | grep -E "^3\.(9|10|11|12)" > /dev/null; then
    echo "Warning: Python 3.9, 3.10, 3.11, or 3.12 is recommended."
    echo "Current version: $PYTHON_VERSION"
fi
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ Pip upgraded"
echo ""

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Check for rkllm_toolkit
echo ""
echo "========================================"
echo "rkllm_toolkit Installation"
echo "========================================"
echo ""
echo "The rkllm_toolkit package is not available on PyPI."
echo "Please download it from the RKLLM GitHub repository:"
echo ""
echo "1. Visit: https://github.com/airockchip/rknn-llm"
echo "2. Navigate to: rkllm-toolkit/packages/"
echo "3. Download the appropriate .whl file for your Python version:"
echo "   - rkllm_toolkit-1.2.3-cp39-cp39-linux_x86_64.whl"
echo "   - rkllm_toolkit-1.2.3-cp310-cp310-linux_x86_64.whl"
echo "   - rkllm_toolkit-1.2.3-cp311-cp311-linux_x86_64.whl"
echo "   - rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl"
echo ""
echo "4. Install it with:"
echo "   pip install path/to/rkllm_toolkit-*.whl"
echo ""
echo "Or place the .whl file in this directory and run:"
echo "   pip install rkllm_toolkit-*.whl"
echo ""

# Check if rkllm_toolkit is already installed
if python -c "import rkllm" 2>/dev/null; then
    echo "✓ rkllm_toolkit is already installed"
else
    echo "⚠ rkllm_toolkit is not installed yet"
    echo "Please install it to use the converter."
fi
echo ""

echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To use the converter, run:"
echo "  python -m src --help"
echo ""
