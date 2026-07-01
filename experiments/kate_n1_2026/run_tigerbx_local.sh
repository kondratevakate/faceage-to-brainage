#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years}"
INPUT_MANIFEST="${INPUT_MANIFEST:-$EXP_DIR/asian_mri_tools_inputs.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/reprocessed_2026/asian_mri_tools/tigerbx}"
PYTHON="${PYTHON:-python3}"
RUN_BX="${RUN_BX:-1}"
RUN_HLC="${RUN_HLC:-1}"
TIGERBX_FLAGS="${TIGERBX_FLAGS:-bmadq}"
TIGERBX_HLC_SAVE="${TIGERBX_HLC_SAVE:-all}"
TIGERBX_USE_GPU="${TIGERBX_USE_GPU:-0}"
TIGERBX_MODEL_DIR="${TIGERBX_MODEL_DIR:-$OUTPUT_ROOT/model_cache}"
MIN_FREE_GB="${MIN_FREE_GB:-50}"
ALLOW_LOW_DISK="${ALLOW_LOW_DISK:-0}"
BLOCK_IF_FS8_RUNNING="${BLOCK_IF_FS8_RUNNING:-1}"

INPUT_DIR="$OUTPUT_ROOT/inputs"
RESOLVED="$OUTPUT_ROOT/tigerbx_inputs_resolved.csv"
SUMMARY_DIR="$OUTPUT_ROOT/summary"
BX_OUT="$OUTPUT_ROOT/bx"
HLC_OUT="$OUTPUT_ROOT/hlc"

if [[ "$BLOCK_IF_FS8_RUNNING" == "1" ]] && pgrep -af "2024_cross_probe|mris_fix_topology" >/dev/null; then
  echo "Refusing TIGERBx run while FS8 2024_cross_probe/topology correction is active." >&2
  echo "Set BLOCK_IF_FS8_RUNNING=0 only if you intentionally want concurrent CPU load." >&2
  exit 3
fi

free_gb="$(df -BG "$DATA_ROOT" | awk 'NR==2 {gsub("G", "", $4); print $4}')"
if [[ "$ALLOW_LOW_DISK" != "1" && "$free_gb" -lt "$MIN_FREE_GB" ]]; then
  echo "Refusing TIGERBx run: only ${free_gb}GB free at DATA_ROOT; require ${MIN_FREE_GB}GB." >&2
  echo "Set ALLOW_LOW_DISK=1 after deciding the disk risk is acceptable." >&2
  exit 3
fi

if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import tigerbx
PY
then
  echo "TigerBx is not importable in this Python environment." >&2
  echo "Suggested CPU setup:" >&2
  echo "  python3 -m venv .venv_tigerbx" >&2
  echo "  source .venv_tigerbx/bin/activate" >&2
  echo "  pip install --no-cache-dir \"tigerbx[cpu] @ https://github.com/htylab/tigerbx/archive/refs/tags/v0.2.3.tar.gz\"" >&2
  exit 2
fi

mkdir -p "$INPUT_DIR" "$SUMMARY_DIR" "$BX_OUT" "$HLC_OUT" "$TIGERBX_MODEL_DIR"

"$PYTHON" "$EXP_DIR/prepare_candidate_inputs.py" \
  --input-manifest "$INPUT_MANIFEST" \
  --data-root "$DATA_ROOT" \
  --candidate-column tigerbx_candidate \
  --output-manifest "$RESOLVED" \
  --materialize-dir "$INPUT_DIR" \
  --materialize symlink \
  --hash

gpu_flag=()
if [[ "$TIGERBX_USE_GPU" == "1" ]]; then
  gpu_flag=(-g)
fi

export TIGERBX_MODEL_DIR

if [[ "$RUN_BX" == "1" ]]; then
  bx_flags=("-${TIGERBX_FLAGS}")
  if [[ "$TIGERBX_USE_GPU" == "1" ]]; then
    bx_flags+=("-g")
  fi
  tiger bx "$INPUT_DIR" "${bx_flags[@]}" -o "$BX_OUT" --continue --verbose 1

  # TIGERBx 0.2.3 does not include QC logs in its --continue existence check.
  # If q was added after an earlier bmad run, regenerate only BET mask + QC.
  if [[ "$TIGERBX_FLAGS" == *q* ]]; then
    input_count="$(find -L "$INPUT_DIR" -maxdepth 1 -type f \( -name "*.nii" -o -name "*.nii.gz" \) | wc -l)"
    qc_count="$(find "$BX_OUT" -maxdepth 1 -type f -name "*_qc-*.log" | wc -l)"
    if [[ "$qc_count" -lt "$input_count" ]]; then
      qc_flags=(-mq)
      if [[ "$TIGERBX_USE_GPU" == "1" ]]; then
        qc_flags+=("-g")
      fi
      tiger bx "$INPUT_DIR" "${qc_flags[@]}" -o "$BX_OUT" --verbose 1
    fi
  fi
fi

if [[ "$RUN_HLC" == "1" ]]; then
  tiger hlc "$INPUT_DIR" --save "$TIGERBX_HLC_SAVE" "${gpu_flag[@]}" -o "$HLC_OUT" --verbose 1
fi

patterns=()
if [[ -d "$BX_OUT" ]]; then
  patterns+=(--input-glob "$BX_OUT/*_aseg.nii.gz")
  patterns+=(--input-glob "$BX_OUT/*_dgm.nii.gz")
  patterns+=(--input-glob "$BX_OUT/*_syn.nii.gz")
fi
if [[ -d "$HLC_OUT" ]]; then
  patterns+=(--input-glob "$HLC_OUT/*_hlc.nii.gz")
fi

if (( ${#patterns[@]} > 0 )); then
  "$PYTHON" "$EXP_DIR/extract_label_volumes.py" \
    "${patterns[@]}" \
    --method tigerbx \
    --output-csv "$SUMMARY_DIR/tigerbx_label_volumes.csv"
fi

cat > "$SUMMARY_DIR/tigerbx_run_metadata.json" <<JSON
{
  "method": "TIGERBx",
  "input_manifest": "$RESOLVED",
  "output_root": "$OUTPUT_ROOT",
  "tigerbx_flags": "$TIGERBX_FLAGS",
  "hlc_save": "$TIGERBX_HLC_SAVE",
  "model_dir": "$TIGERBX_MODEL_DIR",
  "used_gpu": "$TIGERBX_USE_GPU"
}
JSON

echo "TIGERBx outputs written under: $OUTPUT_ROOT"
