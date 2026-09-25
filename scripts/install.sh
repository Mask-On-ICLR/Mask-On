#!/usr/bin/env bash
# Install PyTorch and package dependencies in the active environment.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
