#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -r requirements-scoring.txt
python -m nltk.downloader punkt_tab
command -v bwrap >/dev/null || { echo 'Install bubblewrap using your system administrator/package manager for MBPP.' >&2; exit 1; }
