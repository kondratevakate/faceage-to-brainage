#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NEUROFM_REPO="${NEUROFM_REPO:-/mnt/d/projects/02_academia/_external/NeuroFM}"
NEUROFM_PYTHON="${NEUROFM_PYTHON:-/home/kate/.venvs/neurofm_py311/bin/python}"
NEUROFM_WEIGHTS="${NEUROFM_WEIGHTS:-$NEUROFM_REPO/.cache/neurofm-s.h5}"
EXPECTED_COMMIT="d4e3c463910d939a681d24ebdeb26d44dea6878f"
EXPECTED_WEIGHTS_SHA256="8015a0552214b87e43b5462b6c183f8d0da2d957d7ae11ed09a2e3355f5e991f"
EXPECTED_REMOTE="https://github.com/rockNroll87q/NeuroFM.git"

PREPROCESS_ROOT="${PREPROCESS_ROOT:-/mnt/d/data/faceage-to-brainage/derivatives/hdbet/2.0.1/maclaren_ds000239/R1.0.1}"
PERTURBATION_ROOT="${PERTURBATION_ROOT:-/mnt/d/data/faceage-to-brainage/derivatives/perturbations/1.0/maclaren_ds000239/R1.0.1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/d/data/faceage-to-brainage/derivatives/neurofm/d4e3c46/neurofm-s/maclaren_ds000239/R1.0.1/perturbations}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/brainage}"

if [[ "$(git -C "$NEUROFM_REPO" config --get remote.origin.url)" != "$EXPECTED_REMOTE" ]]; then
  echo "Refusing run: NeuroFM remote mismatch" >&2
  exit 2
fi
if [[ "$(git -C "$NEUROFM_REPO" rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "Refusing run: NeuroFM commit mismatch" >&2
  exit 2
fi
if [[ ! -x "$NEUROFM_PYTHON" || ! -s "$NEUROFM_WEIGHTS" ]]; then
  echo "Missing isolated NeuroFM Python or NeuroFM-S weights" >&2
  exit 2
fi
if [[ "$(sha256sum "$NEUROFM_WEIGHTS" | awk '{print $1}')" != "$EXPECTED_WEIGHTS_SHA256" ]]; then
  echo "Refusing run: NeuroFM-S weight SHA-256 mismatch" >&2
  exit 2
fi
if [[ ! -s "$PREPROCESS_ROOT/neurofm_inputs.csv" ]]; then
  echo "Missing NeuroFM input manifest" >&2
  exit 3
fi
input_count="$("$NEUROFM_PYTHON" -c 'import csv, sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"))))' "$PREPROCESS_ROOT/neurofm_inputs.csv")"
if [[ "$input_count" -ne 120 ]]; then
  echo "Refusing perturbations before complete 120-scan baseline preprocessing" >&2
  exit 3
fi

mkdir -p "$OUTPUT_ROOT" "$DATA_DIR"
"$NEUROFM_PYTHON" "$SCRIPT_DIR/prepare_maclaren_perturbations.py" \
  --preprocessing-status "$PREPROCESS_ROOT/preprocessing_status.csv" \
  --output-root "$PERTURBATION_ROOT"

"$NEUROFM_PYTHON" "$NEUROFM_REPO/scripts/run_inference.py" \
  --input "$PERTURBATION_ROOT/perturbation_inputs.csv" \
  --output "$OUTPUT_ROOT" \
  --model neurofm-s \
  --outputs brain_health,latent \
  --output-mode summary \
  --device cpu \
  --weights "$NEUROFM_WEIGHTS" \
  --cache-dir "$NEUROFM_REPO/.cache" \
  --overwrite

"$NEUROFM_PYTHON" "$SCRIPT_DIR/summarize_maclaren_perturbations.py" \
  --input-manifest "$PERTURBATION_ROOT/perturbation_inputs.csv" \
  --results-summary "$OUTPUT_ROOT/results_summary.csv" \
  --latent-array "$OUTPUT_ROOT/latent_embeddings.npy" \
  --latent-index "$OUTPUT_ROOT/latent_embeddings_index.csv" \
  --baseline-predictions "$DATA_DIR/maclaren_neurofm_predictions.csv" \
  --output-dir "$DATA_DIR"

echo "Maclaren perturbation compact results: $DATA_DIR"
