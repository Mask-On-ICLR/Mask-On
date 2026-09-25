#!/usr/bin/env bash
set -euo pipefail
mask-on download --models "${1:?Usage: download.sh MODEL}" --tasks ${TASKS:-gsm8k mbpp arc_challenge ifeval}
