#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years}"
BRAINFM_REPO="${BRAINFM_REPO:-/mnt/d/projects/02_academia/_external/BrainFM}"
BRAINFM_CHECKPOINT="${BRAINFM_CHECKPOINT:-$BRAINFM_REPO/ckp/brainfm_pretrained.pth}"
INPUT_MANIFEST="${INPUT_MANIFEST:-$EXP_DIR/foundation_model_inputs.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/reprocessed_2026/foundation_models/brainfm}"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-auto}"
WRITE_VOLUMES="${WRITE_VOLUMES:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"
ALLOW_LOW_DISK="${ALLOW_LOW_DISK:-0}"

RESOLVED="$OUTPUT_ROOT/brainfm_inputs_resolved.csv"

if [[ ! -d "$BRAINFM_REPO" ]]; then
  echo "Missing BrainFM repo: $BRAINFM_REPO" >&2
  echo "Clone: git clone https://github.com/jhuldr/BrainFM.git $BRAINFM_REPO" >&2
  exit 2
fi

if [[ ! -f "$BRAINFM_CHECKPOINT" ]]; then
  echo "Missing BrainFM checkpoint: $BRAINFM_CHECKPOINT" >&2
  echo "Download the official BrainFM pretrained weight, place it at ckp/brainfm_pretrained.pth, or set BRAINFM_CHECKPOINT." >&2
  echo "Official model card: https://huggingface.co/peirong26/BrainFM" >&2
  exit 2
fi

free_gb="$(df -BG "$DATA_ROOT" | awk 'NR==2 {gsub("G", "", $4); print $4}')"
if [[ "$ALLOW_LOW_DISK" != "1" && "$free_gb" -lt "$MIN_FREE_GB" ]]; then
  echo "Refusing BrainFM run: only ${free_gb}GB free at DATA_ROOT; require ${MIN_FREE_GB}GB." >&2
  echo "Set ALLOW_LOW_DISK=1 to override after clearing space/QC risk." >&2
  exit 3
fi

mkdir -p "$OUTPUT_ROOT"

"$PYTHON" "$EXP_DIR/prepare_foundation_model_inputs.py" \
  --input-manifest "$INPUT_MANIFEST" \
  --data-root "$DATA_ROOT" \
  --method brainfm \
  --output-manifest "$RESOLVED" \
  --materialize none \
  --hash

volume_flag=()
if [[ "$WRITE_VOLUMES" == "1" ]]; then
  volume_flag+=(--write-volumes)
fi

"$PYTHON" "$EXP_DIR/brainfm_infer_kate.py" \
  --brainfm-repo "$BRAINFM_REPO" \
  --checkpoint "$BRAINFM_CHECKPOINT" \
  --input-manifest "$RESOLVED" \
  --output-dir "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  "${volume_flag[@]}"

echo "BrainFM outputs written under: $OUTPUT_ROOT"
