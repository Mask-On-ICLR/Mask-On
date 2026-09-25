#!/usr/bin/env bash
# Install vLLM and AR evaluation dependencies in the active environment.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install 'vllm==0.10.2' 'datasets==3.6.0' 'PyYAML==6.0.2'
python -m pip install --no-deps -e .
