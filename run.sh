#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="llm_env"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install torch transformers datasets

python main.py
