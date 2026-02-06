#!/usr/bin/env bash
# Run the TikTok-style video generator (SDXL + Wan 2.2).
# Usage: ./run_video.sh "Your prompt"
#        ./run_video.sh --example 2 --preset tiktok_10s
#        ./run_video.sh --list-presets
set -euo pipefail

VENV_DIR="llm_env"

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv for faster package installation..."
    export UV_CACHE_DIR="${VENV_DIR}/.uv_cache"
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

# Allow MPS to use more unified memory (avoids "MPS backend out of memory" on Apple Silicon).
# If you still hit OOM, use: ./run_video.sh --low-memory "Your prompt"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Pass all arguments through to video_generator.py (SDXL image gen + Wan 2.2 video)
python video_generator.py "$@"
