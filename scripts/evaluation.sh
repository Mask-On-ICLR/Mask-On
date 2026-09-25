#!/usr/bin/env bash
set -euo pipefail
model=${1:?Usage: evaluation.sh MODEL [evaluate options]}
shift
for task in ${TASKS:-gsm8k mbpp arc_challenge ifeval}; do
  root="${OUTPUT_ROOT:-outputs}/$model/${MODE:-diffusion}/$task"
  mask-on evaluate --model "$model" --mode "${MODE:-diffusion}" --task "$task" --output "$root" "$@"
  mask-on score "$root"
done
