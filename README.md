# M4 GPU Demo

Run HuggingFace models on Apple Silicon M4 GPU.

## Quick Start

```bash
# Web UI (recommended)
./run_ui.sh

# Power metrics (Avg/Min/Max W) require root
sudo ./run_ui.sh

# Command line
./run.sh
```

## Features

- Automatic GPU detection (MPS backend)
- GPU power monitoring dashboard (samples/duration always; power metrics with `sudo ./run_ui.sh`)
- Hallucination evaluation demo using Vectara model

## Requirements

- macOS with Apple Silicon (M4)
- Python 3.10+

## Manual Setup

```bash
python3 -m venv llm_env
source llm_env/bin/activate
pip install -r requirements.txt
python server.py
```

## Author

Hyun Jae Moon - calhyunjaemoon@gmail.com
