# Mask-On

Compact, mode-selectable AR/diffusion language models using ternary task deltas.

Models: **Fast dLLM v2 · WeDLM-8B · SDAR-8B · LLaDA2.0-mini**  
Benchmarks: **GSM8K · MBPP · ARC-Challenge · IFEval**

## Installation

Python 3.11 and a CUDA-capable GPU are required for model execution.

```bash
conda env create -f environment.yml
conda activate mask-on
bash scripts/install.sh
bash scripts/install_scoring.sh
```

The default runtime uses PyTorch 2.7.1/CUDA 12.8 and Transformers 4.53.1.
MBPP scoring requires `bubblewrap` and enabled user namespaces; generated code
is never executed without sandbox isolation.

WeDLM and AR vLLM use separate environments:

```bash
conda create -n mask-on-wedlm python=3.11 pip -y
conda activate mask-on-wedlm
bash scripts/install_wedlm.sh  # Builds flash-attn using the CUDA toolkit.

conda create -n mask-on-ar python=3.11 pip -y
conda activate mask-on-ar
bash scripts/install_vllm.sh
```

`HF_HOME` selects the Hugging Face cache; `MASK_ON_HOME` selects downloaded
datasets and native decoder sources (default: `./artifacts`).
Authenticate with Hugging Face separately if needed. Never commit credentials.
Model loading executes trusted, revision-pinned checkpoint code.

## Usage

### Download

```bash
mask-on list
mask-on download --models fast --tasks gsm8k mbpp arc_challenge ifeval --dry-run
mask-on download --models fast --tasks gsm8k mbpp arc_challenge ifeval
# Other model names: wedlm, sdar, llada.
mask-on native-source sdar
```

Model, dataset and decoder revisions are pinned in
[`src/mask_on/catalog.json`](src/mask_on/catalog.json).
Use `--metadata-only` to download tokenizer/config/code without weights.

### Prepare Mask-On

```bash
conda activate mask-on
# Nemotron: 32 fit + 8 validation items per code/math/stem/chat domain.
mask-on calibration-panel --output artifacts/panel
CUDA_VISIBLE_DEVICES=0 mask-on collect --model fast \
  --panel artifacts/panel/panel.json --output artifacts/fast-activation
mask-on fit --model fast --collection artifacts/fast-activation \
  --output artifacts/fast-fit --device cpu
mask-on prepare --model fast --method mask_on --state artifacts/fast-fit \
  --output artifacts/fast-mask-on
```

Collection saves endpoint completions and projection inputs from clean-sequence
forwards. Fitting uses full projection-output least squares with nonnegative
output-channel scales and searches `alpha = 0.05, 0.10, ..., 0.95` using fit
error only. Use `fit --alpha 0.4` for a fixed-support comparison.
Keep the same saved calibration panel across compared methods.
`payload/` contains compact codes/scales and endpoint vocabulary tensors;
`checkpoint/` is the dense evaluation checkpoint.

### Baselines

```bash
mask-on prepare --model fast --method task_arithmetic --beta 0.4 --output artifacts/fast-ta
mask-on prepare --model fast --method ties --output artifacts/fast-ties
mask-on prepare --model fast --method dare --output artifacts/fast-dare
mask-on prepare --model fast --method t_switch --output artifacts/fast-tswitch
```

Defaults are in [`configs/baselines.yaml`](configs/baselines.yaml).
AdaMerging/++, AIM, RegMean, BitDelta and Delta-CoMe consume fitted state via
`prepare --state DIR`. Fitting/statistics APIs live in `src/mask_on/merge/`
and `src/mask_on/evaluation/`; automated native collection/training commands
are not available for every baseline. Representation Surgery supplies native
adapter APIs, not a flattened checkpoint or a generic vLLM adapter.

<details>
<summary>Fitted-state format</summary>

Each directory contains `manifest.json` and a `DONE.json` map of file SHA256
hashes. The manifest binds model, method, checkpoint revisions, calibration,
recipe and coverage. `base_files` and `endpoint_files` contain inventories
from `mask_on.checkpoints.identity`. Export with
`mask_on.artifacts.atomic_json` and `finish` after checking completion.

- Mask-On: `selection.json` and `scales.safetensors`, emitted by `mask-on fit`.
- AdaMerging/++: `coefficients.json` keyed by parameter, with selection
  `minimum_validation_loss_earliest_tie`; audit using `audit_completed_fit`.
- AIM: `importance.safetensors`; mean absolute input-channel activation
  normalized by each module's maximum.
- RegMean: `grams.safetensors`, keyed `ar::<parameter>` and
  `diffusion::<parameter>`; row-normalized, uncentered full Grams.
  Any numerical regularization must be explicit.
- BitDelta/Delta-CoMe: a `compressed/` bundle from `write_payload`, including
  `receipt.json`, `payload.safetensors` and `DONE.json`. Bind model, method
  and checkpoint identities and include all files in the outer file map.

No method substitutes guessed coefficients or silently fills missing statistics.
Prepare once, then reuse the frozen state across benchmarks and modes.

</details>

### Evaluate

```bash
# Generate two GSM8K responses; omit --checkpoint for the diffusion endpoint.
CUDA_VISIBLE_DEVICES=0 mask-on evaluate --model fast --task gsm8k --limit 2 \
  --checkpoint artifacts/fast-mask-on/checkpoint --output outputs/fast-check

CUDA_VISIBLE_DEVICES=0 OUTPUT_ROOT=outputs/mask-on \
  TASKS="gsm8k mbpp arc_challenge ifeval" \
  bash scripts/evaluation.sh fast --checkpoint artifacts/fast-mask-on/checkpoint

# Score saved generations on CPU.
mask-on score outputs/mask-on/fast/diffusion/mbpp
mask-on results outputs/mask-on/fast/diffusion/{gsm8k,mbpp,arc_challenge,ifeval} \
  --csv outputs/results.csv
```

Each task has its own manifest, per-item outputs, metrics and completion marker.
Identical interrupted commands reuse saved items. Changing a checkpoint or
protocol requires a new output directory.

AR evaluation uses vLLM with matched benchmark prompts, greedy budgets and
scorers. Qwen3 runs in non-thinking mode.

```bash
conda activate mask-on-ar
CUDA_VISIBLE_DEVICES=0 mask-on evaluate --model fast --mode ar --backend vllm \
  --task gsm8k --output outputs/fast-ar-endpoint

# Write a native causal configuration for the merged weights.
mask-on ar-view --model fast --checkpoint artifacts/fast-ta/checkpoint \
  --output artifacts/fast-ta-ar
CUDA_VISIBLE_DEVICES=0 mask-on evaluate --model fast --mode ar --backend vllm \
  --task gsm8k --checkpoint artifacts/fast-ta-ar/checkpoint --output outputs/fast-ta-ar
```

### Draft-and-review

```bash
conda activate mask-on
CUDA_VISIBLE_DEVICES=0 mask-on draft-review \
  --checkpoint artifacts/fast-mask-on/checkpoint \
  --threshold 0.9 --output outputs/fast-review09
```

Use `--threshold 0.8` and a separate output directory for the other setting.
Each GSM8K item saves its diffusion draft before a fresh native Qwen AR review;
saved drafts survive review failures. Metrics separate diffusion accuracy/TPS,
reviewed accuracy, answer changes and combined generation TPS.
This implementation keeps two dense states; it does not subtract rounded
weights or claim lossless switching. Run a small GPU check before full evaluation.
