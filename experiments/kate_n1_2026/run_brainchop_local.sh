#!/usr/bin/env bash
set -euo pipefail

VERSION="${BRAINCHOP_VERSION:-0.2.5}"
VENV="${BRAINCHOP_VENV:-$HOME/.venvs/brainchop}"
DATA_ROOT="${DATA_ROOT:-/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${BRAINCHOP_MANIFEST:-$SCRIPT_DIR/brainchop_inputs.csv}"
OUTPUT_ROOT="${BRAINCHOP_OUTPUT_ROOT:-$DATA_ROOT/reprocessed_2026/brainchop/brainchop_${VERSION}}"
MODELS="${BRAINCHOP_MODELS:-subcortical}"
SCAN_IDS="${BRAINCHOP_SCAN_IDS:-}"
TIMEOUT_SECONDS="${BRAINCHOP_TIMEOUT_SECONDS:-1800}"
LIMIT="${BRAINCHOP_LIMIT:-0}"

python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
pip install "brainchop==$VERSION"

if ! command -v clang >/dev/null 2>&1; then
  echo "ERROR: BrainChop/tinygrad CPU inference requires clang. Install with: sudo apt-get install -y clang" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
brainchop --list | tee "$OUTPUT_ROOT/brainchop_models.txt"

python "$SCRIPT_DIR/run_brainchop_batch.py" \
  --manifest "$MANIFEST" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --brainchop-bin "$VENV/bin/brainchop" \
  --models "$MODELS" \
  --scan-ids "$SCAN_IDS" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --limit "$LIMIT"
