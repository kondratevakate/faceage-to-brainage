#!/usr/bin/env bash
set -euo pipefail

# NeuroFM SIMON all-orig diagnostic branch. This converts existing FastSurfer
# `_orig.mgz` internal source images to NIfTI and runs NeuroFM without skull
# stripping. It is a non-conforming input stress test, not validation.

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/kate_n1_2026}"
NEUROFM_PYTHON="${NEUROFM_PYTHON:-/home/kate/.venvs/neurofm_py311/bin/python}"
export NEUROFM_PYTHON

RAW_OUTPUT_ROOT="${RAW_OUTPUT_ROOT:-$DATA_ROOT/reprocessed_2026/foundation_models/neurofm_simon_raw_orig}"
RAW_NIFTI_DIR="${RAW_NIFTI_DIR:-$RAW_OUTPUT_ROOT/raw_orig_nifti_inputs}"
RAW_INPUT_STATUS_CSV="${RAW_INPUT_STATUS_CSV:-$DATA_DIR/neurofm_simon_raw_orig_input_status.csv}"
RAW_INPUTS_RESOLVED="${RAW_INPUTS_RESOLVED:-$DATA_DIR/neurofm_simon_raw_orig_inputs_resolved.csv}"
SOURCE_ORIG_MANIFEST="${SOURCE_ORIG_MANIFEST:-$EXP_DIR/midi_brainage_simon_all_orig_inputs.csv}"
PREPARE_RAW_ORIG_INPUTS="${PREPARE_RAW_ORIG_INPUTS:-1}"
OVERWRITE_RAW_ORIG_INPUTS="${OVERWRITE_RAW_ORIG_INPUTS:-0}"

if [[ "$PREPARE_RAW_ORIG_INPUTS" == "1" ]]; then
  raw_prep_flags=()
  if [[ "$OVERWRITE_RAW_ORIG_INPUTS" == "1" ]]; then
    raw_prep_flags+=(--overwrite)
  fi
  "$NEUROFM_PYTHON" "$EXP_DIR/prepare_neurofm_orig_nifti_inputs.py" \
    --input-manifest "$SOURCE_ORIG_MANIFEST" \
    --output-dir "$RAW_NIFTI_DIR" \
    --output-manifest "$RAW_INPUTS_RESOLVED" \
    --status-csv "$RAW_INPUT_STATUS_CSV" \
    --limit "${LIMIT:-0}" \
    "${raw_prep_flags[@]}"
fi

export SOURCE_MANIFEST="${SOURCE_MANIFEST:-$RAW_INPUTS_RESOLVED}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$RAW_OUTPUT_ROOT}"
export RESULTS_DIR="${RESULTS_DIR:-$RAW_OUTPUT_ROOT/results}"
export PREDICTIONS_CSV="${PREDICTIONS_CSV:-$DATA_DIR/neurofm_simon_raw_orig_predictions.csv}"
export SUMMARY_CSV="${SUMMARY_CSV:-$DATA_DIR/neurofm_simon_raw_orig_summary.csv}"
export METADATA_JSON="${METADATA_JSON:-$DATA_DIR/neurofm_simon_raw_orig_metadata.json}"
export PREPARE_MASKED_INPUTS=0
export MODEL_VARIANT="${MODEL_VARIANT:-neurofm-s}"
export NEUROFM_OUTPUTS="${NEUROFM_OUTPUTS:-brain_health}"
export DEVICE="${DEVICE:-cpu}"
export NEUROFM_INTERPRETATION="${NEUROFM_INTERPRETATION:-NeuroFM run on non-skull-stripped FastSurfer orig.mgz images converted to NIfTI. This is an all-orig diagnostic stress branch, not a validation claim or calibrated brain-age estimate.}"
export NEUROFM_WEIGHTS_SOURCE="${NEUROFM_WEIGHTS_SOURCE:-https://huggingface.co/NeuroAI-UofG/NeuroFM, neurofm-s.h5, downloaded after gated access acceptance on 2026-07-09; kept outside git.}"

bash "$EXP_DIR/run_neurofm_local.sh"
