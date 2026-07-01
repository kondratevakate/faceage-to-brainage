#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years}"
BRAINIAC_REPO="${BRAINIAC_REPO:-/mnt/d/projects/02_academia/_external/BrainIAC}"
BRAINIAC_CHECKPOINT="${BRAINIAC_CHECKPOINT:-$BRAINIAC_REPO/src/checkpoints/BrainIAC.ckpt}"
BRAINIAC_TEMPLATE="${BRAINIAC_TEMPLATE:-$BRAINIAC_REPO/src/preprocessing/atlases/nihpd_asym_13.0-18.5_t1w.nii}"
INPUT_MANIFEST="${INPUT_MANIFEST:-$EXP_DIR/foundation_model_inputs.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/reprocessed_2026/foundation_models/brainiac}"
PYTHON="${PYTHON:-python3}"
RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
DEVICE="${DEVICE:-auto}"

RAW_DIR="$OUTPUT_ROOT/raw_inputs"
PRE_DIR="$OUTPUT_ROOT/preprocessed"
FEATURE_DIR="$OUTPUT_ROOT/features"
RESOLVED_RAW="$OUTPUT_ROOT/brainiac_inputs_resolved.csv"

if [[ ! -d "$BRAINIAC_REPO/src" ]]; then
  echo "Missing BrainIAC repo: $BRAINIAC_REPO" >&2
  echo "Clone: git clone https://github.com/AIM-KannLab/BrainIAC.git $BRAINIAC_REPO" >&2
  exit 2
fi

if [[ ! -f "$BRAINIAC_CHECKPOINT" ]]; then
  echo "Missing BrainIAC checkpoint: $BRAINIAC_CHECKPOINT" >&2
  echo "Use the official GitHub Dropbox checkpoint or HF backbone.safetensors, then set BRAINIAC_CHECKPOINT." >&2
  echo "HF model card: https://huggingface.co/eugenehp/brainiac" >&2
  exit 2
fi

mkdir -p "$RAW_DIR" "$PRE_DIR" "$FEATURE_DIR"

"$PYTHON" "$EXP_DIR/prepare_foundation_model_inputs.py" \
  --input-manifest "$INPUT_MANIFEST" \
  --data-root "$DATA_ROOT" \
  --method brainiac \
  --output-manifest "$RESOLVED_RAW" \
  --materialize-dir "$RAW_DIR" \
  --materialize symlink \
  --hash

if [[ "$RUN_PREPROCESS" == "1" ]]; then
  if [[ ! -f "$BRAINIAC_TEMPLATE" ]]; then
    echo "Missing BrainIAC preprocessing template: $BRAINIAC_TEMPLATE" >&2
    exit 2
  fi
  PYTHONPATH="$BRAINIAC_REPO/src/preprocessing:$BRAINIAC_REPO/src:${PYTHONPATH:-}" \
    "$PYTHON" "$BRAINIAC_REPO/src/preprocessing/mri_preprocess_3d_simple.py" \
      --temp_img "$BRAINIAC_TEMPLATE" \
      --input_dir "$RAW_DIR" \
      --output_dir "$PRE_DIR"
fi

"$PYTHON" "$EXP_DIR/extract_brainiac_embeddings.py" \
  --checkpoint "$BRAINIAC_CHECKPOINT" \
  --brainiac-repo "$BRAINIAC_REPO" \
  --image-dir "$PRE_DIR" \
  --output-csv "$FEATURE_DIR/brainiac_embeddings.csv" \
  --output-metadata "$FEATURE_DIR/brainiac_embeddings_metadata.json" \
  --device "$DEVICE"

echo "BrainIAC features written under: $FEATURE_DIR"
