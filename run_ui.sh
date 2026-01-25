#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="llm_env"

if command -v uv &> /dev/null; then
    echo "Using uv..."
    [ ! -d "$VENV_DIR" ] && uv venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    uv pip install -r requirements.txt
else
    echo "Using pip..."
    [ ! -d "$VENV_DIR" ] && python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install -q -r requirements.txt
fi

# Power metrics (Avg/Min/Max W) require root; run with: sudo ./run_ui.sh
python server.py
