#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NEUROFM_REPO="${NEUROFM_REPO:-/mnt/d/projects/02_academia/_external/NeuroFM}"
NEUROFM_PYTHON="${NEUROFM_PYTHON:-/home/kate/.venvs/neurofm_py311/bin/python}"
NEUROFM_WEIGHTS="${NEUROFM_WEIGHTS:-$NEUROFM_REPO/.cache/neurofm-s.h5}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-d4e3c463910d939a681d24ebdeb26d44dea6878f}"
EXPECTED_WEIGHTS_SHA256="${EXPECTED_WEIGHTS_SHA256:-8015a0552214b87e43b5462b6c183f8d0da2d957d7ae11ed09a2e3355f5e991f}"

PREPROCESS_ROOT="${PREPROCESS_ROOT:-/mnt/d/data/faceage-to-brainage/derivatives/hdbet/2.0.1/maclaren_ds000239/R1.0.1}"
INPUT_MANIFEST="${INPUT_MANIFEST:-$PREPROCESS_ROOT/neurofm_inputs.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/d/data/faceage-to-brainage/derivatives/neurofm/d4e3c46/neurofm-s/maclaren_ds000239/R1.0.1}"
RESULTS_DIR="$OUTPUT_ROOT/baseline_results"
SCHEMA_DIR="$OUTPUT_ROOT/schema_test"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/brainage}"

remote_url="$(git -C "$NEUROFM_REPO" config --get remote.origin.url)"
if [[ "$remote_url" != "https://github.com/rockNroll87q/NeuroFM.git" && "$remote_url" != "git@github.com:rockNroll87q/NeuroFM.git" ]]; then
  echo "Refusing run: unexpected NeuroFM remote: $remote_url" >&2
  exit 2
fi

actual_commit="$(git -C "$NEUROFM_REPO" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "Refusing run: NeuroFM commit $actual_commit != $EXPECTED_COMMIT" >&2
  exit 2
fi

if [[ ! -x "$NEUROFM_PYTHON" ]]; then
  echo "Missing isolated NeuroFM Python: $NEUROFM_PYTHON" >&2
  exit 2
fi
if [[ ! -s "$NEUROFM_WEIGHTS" ]]; then
  echo "Missing NeuroFM weights: $NEUROFM_WEIGHTS" >&2
  exit 2
fi

actual_weights_sha256="$(sha256sum "$NEUROFM_WEIGHTS" | awk '{print $1}')"
if [[ "$actual_weights_sha256" != "$EXPECTED_WEIGHTS_SHA256" ]]; then
  echo "Refusing run: unexpected NeuroFM-S weight SHA-256: $actual_weights_sha256" >&2
  exit 2
fi

if [[ ! -s "$INPUT_MANIFEST" ]]; then
  echo "Missing HD-BET NeuroFM input manifest: $INPUT_MANIFEST" >&2
  exit 3
fi
input_count="$("$NEUROFM_PYTHON" -c 'import csv, sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"))))' "$INPUT_MANIFEST")"
if [[ "$input_count" -ne 120 ]]; then
  echo "Refusing partial baseline: expected 120 inputs, found $input_count" >&2
  exit 3
fi

mkdir -p "$RESULTS_DIR" "$SCHEMA_DIR" "$DATA_DIR"
first_input="$("$NEUROFM_PYTHON" -c 'import csv, sys; print(next(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))["input"])' "$INPUT_MANIFEST")"
if [[ -z "$first_input" || ! -s "$first_input" ]]; then
  echo "Could not resolve first schema-test input from $INPUT_MANIFEST" >&2
  exit 3
fi

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

"$NEUROFM_PYTHON" "$SCRIPT_DIR/validate_neurofm_schema.py" \
  --neurofm-repo "$NEUROFM_REPO" \
  --weights "$NEUROFM_WEIGHTS" \
  --input "$first_input" \
  --work-dir "$SCHEMA_DIR" \
  --output-json "$SCHEMA_DIR/schema_validation.json"

"$NEUROFM_PYTHON" "$NEUROFM_REPO/scripts/run_inference.py" \
  --input "$INPUT_MANIFEST" \
  --output "$RESULTS_DIR" \
  --model neurofm-s \
  --outputs brain_health,latent \
  --output-mode summary \
  --device cpu \
  --weights "$NEUROFM_WEIGHTS" \
  --cache-dir "$NEUROFM_REPO/.cache" \
  --overwrite

"$NEUROFM_PYTHON" "$SCRIPT_DIR/summarize_maclaren_neurofm.py" \
  --preprocessing-status "$PREPROCESS_ROOT/preprocessing_status.csv" \
  --results-summary "$RESULTS_DIR/results_summary.csv" \
  --latent-array "$RESULTS_DIR/latent_embeddings.npy" \
  --latent-index "$RESULTS_DIR/latent_embeddings_index.csv" \
  --schema-validation "$SCHEMA_DIR/schema_validation.json" \
  --output-dir "$DATA_DIR"

echo "Maclaren NeuroFM compact results: $DATA_DIR"
