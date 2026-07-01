#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years}"
OPENMAP_REPO="${OPENMAP_REPO:-/mnt/d/projects/02_academia/_external/OpenMAP-T1}"
OPENMAP_MODEL_DIR="${OPENMAP_MODEL_DIR:-/mnt/d/projects/02_academia/_models/OpenMAP-T1-v3}"
INPUT_MANIFEST="${INPUT_MANIFEST:-$EXP_DIR/asian_mri_tools_inputs.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/reprocessed_2026/asian_mri_tools/openmap_t1}"
PYTHON="${PYTHON:-python3}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"
ALLOW_LOW_DISK="${ALLOW_LOW_DISK:-0}"
BLOCK_IF_FS8_RUNNING="${BLOCK_IF_FS8_RUNNING:-1}"

INPUT_DIR="$OUTPUT_ROOT/inputs"
RESOLVED="$OUTPUT_ROOT/openmap_t1_inputs_resolved.csv"

if [[ "$BLOCK_IF_FS8_RUNNING" == "1" ]] && pgrep -af "2024_cross_probe|mris_fix_topology" >/dev/null; then
  echo "Refusing OpenMAP-T1 run while FS8 2024_cross_probe/topology correction is active." >&2
  exit 3
fi

free_gb="$(df -BG "$DATA_ROOT" | awk 'NR==2 {gsub("G", "", $4); print $4}')"
if [[ "$ALLOW_LOW_DISK" != "1" && "$free_gb" -lt "$MIN_FREE_GB" ]]; then
  echo "Refusing OpenMAP-T1 run: only ${free_gb}GB free at DATA_ROOT; require ${MIN_FREE_GB}GB." >&2
  exit 3
fi

if [[ ! -f "$OPENMAP_REPO/src/parcellation.py" ]]; then
  echo "Missing OpenMAP-T1 repo: $OPENMAP_REPO" >&2
  echo "Clone: git clone https://github.com/OishiLab/OpenMAP-T1.git $OPENMAP_REPO" >&2
  exit 2
fi

if [[ ! -d "$OPENMAP_MODEL_DIR" ]] || [[ -z "$(find "$OPENMAP_MODEL_DIR" -maxdepth 1 -type f | head -1)" ]]; then
  echo "Missing OpenMAP-T1 pretrained model folder: $OPENMAP_MODEL_DIR" >&2
  echo "The official repo requires applying for/downloading OpenMAP-T1 v3 weights before local inference." >&2
  echo "Model request link is documented in the OpenMAP-T1 README." >&2
  exit 2
fi

mkdir -p "$INPUT_DIR" "$OUTPUT_ROOT"

"$PYTHON" "$EXP_DIR/prepare_candidate_inputs.py" \
  --input-manifest "$INPUT_MANIFEST" \
  --data-root "$DATA_ROOT" \
  --candidate-column openmap_t1_candidate \
  --output-manifest "$RESOLVED" \
  --materialize-dir "$INPUT_DIR" \
  --materialize symlink \
  --hash

cd "$OPENMAP_REPO"
"$PYTHON" src/parcellation.py \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_ROOT/parcellation" \
  -m "$OPENMAP_MODEL_DIR" \
  --output-space native

echo "OpenMAP-T1 outputs written under: $OUTPUT_ROOT"
