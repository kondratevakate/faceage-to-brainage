#!/usr/bin/env bash
set -euo pipefail

# NeuroFM SIMON branch using existing FastSurfer `_orig.mgz` internal source
# images plus matching `aparcDKT+aseg.mgz` labels as a skull-strip surrogate.
# This prepares/runs an application/QC branch only. It is not a validation claim,
# and NeuroFM's documented age range makes younger SIMON sessions a domain-risk
# sanity check rather than clean calibration.

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/kate_n1_2026}"

export SOURCE_MANIFEST="${SOURCE_MANIFEST:-$EXP_DIR/midi_brainage_simon_all_orig_inputs.csv}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/reprocessed_2026/foundation_models/neurofm_simon_fastsurfer_mask}"
export PREDICTIONS_CSV="${PREDICTIONS_CSV:-$DATA_DIR/neurofm_simon_fastsurfer_mask_predictions.csv}"
export SUMMARY_CSV="${SUMMARY_CSV:-$DATA_DIR/neurofm_simon_fastsurfer_mask_summary.csv}"
export METADATA_JSON="${METADATA_JSON:-$DATA_DIR/neurofm_simon_fastsurfer_mask_metadata.json}"
export INPUT_STATUS_CSV="${INPUT_STATUS_CSV:-$OUTPUT_ROOT/neurofm_simon_fastsurfer_mask_input_status.csv}"
export INFER_FASTSURFER_LABEL="${INFER_FASTSURFER_LABEL:-1}"
export PREPARE_MASKED_INPUTS="${PREPARE_MASKED_INPUTS:-1}"
export MODEL_VARIANT="${MODEL_VARIANT:-neurofm-s}"
export NEUROFM_OUTPUTS="${NEUROFM_OUTPUTS:-brain_health}"
export DEVICE="${DEVICE:-cpu}"

bash "$EXP_DIR/run_neurofm_local.sh"
