#!/usr/bin/env bash
set -euo pipefail

# Prepare Kate T1-like skull-stripped inputs for the user-requested NeuroFM
# branch. This creates input artifacts only; it does not run NeuroFM inference.

DATA_ROOT="${DATA_ROOT:-/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/reprocessed_2026/foundation_models/neurofm_hdbet_inputs}"
HD_BET="${HD_BET:-/home/kate/.venvs/midi_brainage_py311/bin/hd-bet}"
DEVICE="${DEVICE:-cpu}"
DISABLE_TTA="${DISABLE_TTA:-1}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -x "$HD_BET" ]]; then
  echo "Missing HD-BET executable: $HD_BET" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR/logs"
batch_log="$OUTPUT_DIR/logs/hdbet_kate_batch.log"

run_case() {
  local scan_id="$1"
  local input_path="$2"
  local output_path="$OUTPUT_DIR/${scan_id}_hdbet.nii.gz"
  local log_path="$OUTPUT_DIR/logs/${scan_id}_hdbet.log"

  echo "[$(date -Is)] START $scan_id input=$input_path output=$output_path" | tee -a "$batch_log"

  if [[ ! -f "$input_path" ]]; then
    echo "[$(date -Is)] FAIL $scan_id missing_input=$input_path" | tee -a "$batch_log"
    return 1
  fi

  if [[ "$OVERWRITE" != "1" && -s "$output_path" ]]; then
    echo "[$(date -Is)] SKIP $scan_id existing=$output_path" | tee -a "$batch_log"
    return 0
  fi

  args=(-i "$input_path" -o "$output_path" -device "$DEVICE" --save_bet_mask)
  if [[ "$DISABLE_TTA" == "1" ]]; then
    args+=(--disable_tta)
  fi

  if "$HD_BET" "${args[@]}" >"$log_path" 2>&1; then
    echo "[$(date -Is)] DONE $scan_id" | tee -a "$batch_log"
  else
    rc=$?
    echo "[$(date -Is)] FAIL $scan_id rc=$rc log=$log_path" | tee -a "$batch_log"
    return "$rc"
  fi
}

run_case kate_2018_t1 "$DATA_ROOT/images/2018/nifti/3_fspgr_bravo_10mm_ax.nii.gz"
run_case kate_2022_t1 "$DATA_ROOT/images/2022/nifti/4_t1_se_sag.nii.gz"
run_case kate_2024_3di "$DATA_ROOT/images/2024/nifti/901_3di_mc_hr.nii.gz"
run_case kate_2024_t1_ffe_401 "$DATA_ROOT/images/2024/nifti/401_t1w_ffe.nii.gz"
run_case kate_2024_t1_ffe_601 "$DATA_ROOT/images/2024/nifti/601_t1w_ffe.nii.gz"
