#!/usr/bin/env bash
# Install the WeDLM runtime in the active Python 3.11 environment.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements-wedlm.txt
python -m pip install --no-deps -e .
mask-on native-source wedlm
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -c 'import subprocess,sys; from mask_on.download import native_source; subprocess.run([sys.executable,"-m","pip","install","--no-deps","-e",str(native_source("wedlm"))],check=True)'
