#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="llm_env"

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv for faster package installation..."
    if [ ! -d "$VENV_DIR" ]; then
        uv venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    uv pip install -r requirements.txt
else
    echo "Using pip (install uv for faster installation: curl -LsSf https://astral.sh/uv/install.sh | sh)"
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip
    pip install -r requirements.txt
fi

python main.py
