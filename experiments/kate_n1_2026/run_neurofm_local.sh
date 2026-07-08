#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years}"
NEUROFM_REPO="${NEUROFM_REPO:-/mnt/d/projects/02_academia/_external/NeuroFM}"
NEUROFM_PYTHON="${NEUROFM_PYTHON:-/home/kate/.venvs/neurofm_py311/bin/python}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$EXP_DIR/neurofm_kate_brainchop_mask_inputs.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/reprocessed_2026/foundation_models/neurofm}"
MASKED_INPUT_DIR="${MASKED_INPUT_DIR:-$OUTPUT_ROOT/masked_inputs}"
INPUT_MANIFEST_RESOLVED="${INPUT_MANIFEST_RESOLVED:-$OUTPUT_ROOT/neurofm_inputs_resolved.csv}"
INPUT_STATUS_CSV="${INPUT_STATUS_CSV:-$OUTPUT_ROOT/neurofm_masked_input_status.csv}"
RESULTS_DIR="${RESULTS_DIR:-$OUTPUT_ROOT/results}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/kate_n1_2026}"
PREDICTIONS_CSV="${PREDICTIONS_CSV:-$DATA_DIR/neurofm_predictions.csv}"
SUMMARY_CSV="${SUMMARY_CSV:-$DATA_DIR/neurofm_summary.csv}"
METADATA_JSON="${METADATA_JSON:-$DATA_DIR/neurofm_metadata.json}"
MODEL_VARIANT="${MODEL_VARIANT:-neurofm-s}"
NEUROFM_OUTPUTS="${NEUROFM_OUTPUTS:-brain_health}"
DEVICE="${DEVICE:-cpu}"
NEUROFM_CACHE_DIR="${NEUROFM_CACHE_DIR:-/mnt/d/projects/02_academia/_external/NeuroFM/.cache}"
NEUROFM_WEIGHTS="${NEUROFM_WEIGHTS:-}"
LIMIT="${LIMIT:-0}"
INFER_FASTSURFER_LABEL="${INFER_FASTSURFER_LABEL:-0}"
OVERWRITE_MASKED_INPUTS="${OVERWRITE_MASKED_INPUTS:-0}"
PREPARE_MASKED_INPUTS="${PREPARE_MASKED_INPUTS:-1}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
ALLOW_LOW_DISK="${ALLOW_LOW_DISK:-0}"

if [[ ! -d "$NEUROFM_REPO/.git" ]]; then
  echo "Missing NeuroFM repo: $NEUROFM_REPO" >&2
  echo "Expected upstream: https://github.com/rockNroll87q/NeuroFM" >&2
  exit 2
fi

remote_url="$(git -C "$NEUROFM_REPO" config --get remote.origin.url)"
if [[ "$remote_url" != "https://github.com/rockNroll87q/NeuroFM.git" && "$remote_url" != "git@github.com:rockNroll87q/NeuroFM.git" ]]; then
  echo "Refusing run: NeuroFM remote is not the user-requested repo: $remote_url" >&2
  exit 2
fi

if [[ ! -x "$NEUROFM_PYTHON" ]]; then
  echo "Missing NeuroFM Python executable: $NEUROFM_PYTHON" >&2
  echo "Create an isolated Python 3.11 venv and install NeuroFM there." >&2
  exit 2
fi

free_gb="$(df -BG "$DATA_ROOT" | awk 'NR==2 {gsub("G", "", $4); print $4}')"
if [[ "$ALLOW_LOW_DISK" != "1" && "$free_gb" -lt "$MIN_FREE_GB" ]]; then
  echo "Refusing NeuroFM run: only ${free_gb}GB free at DATA_ROOT; require ${MIN_FREE_GB}GB." >&2
  exit 3
fi

mkdir -p "$OUTPUT_ROOT" "$RESULTS_DIR" "$DATA_DIR" "$NEUROFM_CACHE_DIR"

prep_flags=()
if [[ "$INFER_FASTSURFER_LABEL" == "1" ]]; then
  prep_flags+=(--infer-fastsurfer-label)
fi
if [[ "$OVERWRITE_MASKED_INPUTS" == "1" ]]; then
  prep_flags+=(--overwrite)
fi

if [[ "$PREPARE_MASKED_INPUTS" == "1" ]]; then
  "$NEUROFM_PYTHON" "$EXP_DIR/prepare_neurofm_masked_inputs.py" \
    --input-manifest "$SOURCE_MANIFEST" \
    --output-dir "$MASKED_INPUT_DIR" \
    --output-manifest "$INPUT_MANIFEST_RESOLVED" \
    --status-csv "$INPUT_STATUS_CSV" \
    --limit "$LIMIT" \
    "${prep_flags[@]}"
else
  INPUT_MANIFEST_RESOLVED="$SOURCE_MANIFEST"
fi

resolved_count="$(awk 'END {print (NR > 0 ? NR - 1 : 0)}' "$INPUT_MANIFEST_RESOLVED")"
if [[ "$resolved_count" -lt 1 ]]; then
  echo "No NeuroFM-ready inputs in: $INPUT_MANIFEST_RESOLVED" >&2
  if [[ "$PREPARE_MASKED_INPUTS" == "1" ]]; then
    echo "See preprocessing status: $INPUT_STATUS_CSV" >&2
  fi
  exit 4
fi

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "Prepared ${resolved_count} NeuroFM-ready input(s): $INPUT_MANIFEST_RESOLVED"
  echo "PREPARE_ONLY=1, skipping NeuroFM model inference."
  exit 0
fi

weights_arg=()
if [[ -n "$NEUROFM_WEIGHTS" ]]; then
  weights_arg+=(--weights "$NEUROFM_WEIGHTS")
fi

"$NEUROFM_PYTHON" "$NEUROFM_REPO/scripts/run_inference.py" \
  --input "$INPUT_MANIFEST_RESOLVED" \
  --output "$RESULTS_DIR" \
  --model "$MODEL_VARIANT" \
  --outputs "$NEUROFM_OUTPUTS" \
  --output-mode summary \
  --device "$DEVICE" \
  --cache-dir "$NEUROFM_CACHE_DIR" \
  --overwrite \
  "${weights_arg[@]}"

source_commit="$(git -C "$NEUROFM_REPO" rev-parse HEAD)"
"$NEUROFM_PYTHON" "$EXP_DIR/summarize_neurofm_results.py" \
  --results-summary "$RESULTS_DIR/results_summary.csv" \
  --input-manifest "$INPUT_MANIFEST_RESOLVED" \
  --output-csv "$PREDICTIONS_CSV" \
  --summary-csv "$SUMMARY_CSV" \
  --metadata-json "$METADATA_JSON" \
  --summary-id "$(basename "$PREDICTIONS_CSV" .csv)" \
  --source-repo "$remote_url" \
  --source-commit "$source_commit" \
  --variant "$MODEL_VARIANT" \
  --weights-cache-dir "${NEUROFM_WEIGHTS:-$NEUROFM_CACHE_DIR}" \
  --interpretation "NeuroFM run on a skull-stripped, mask-derived application branch. This is not validation evidence for Kate brain age or segmentation quality."

echo "NeuroFM predictions: $PREDICTIONS_CSV"
echo "NeuroFM summary: $SUMMARY_CSV"
