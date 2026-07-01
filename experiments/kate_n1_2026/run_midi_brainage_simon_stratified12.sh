#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/kate_n1_2026"

VENV_PYTHON="${MIDI_BRAINAGE_PYTHON:-/home/kate/.venvs/midi_brainage_py311/bin/python}"
WORK_DIR="${MIDI_BRAINAGE_WORK_DIR:-/home/kate/midi_brainage_work}"
MANIFEST="${MIDI_BRAINAGE_MANIFEST:-$EXP_DIR/midi_brainage_simon_stratified12_inputs.csv}"
OUTPUT_CSV="${MIDI_BRAINAGE_OUTPUT_CSV:-$DATA_DIR/midi_brainage_simon_stratified12_predictions.csv}"
SUMMARY_CSV="${MIDI_BRAINAGE_SUMMARY_CSV:-$DATA_DIR/midi_brainage_simon_stratified12_summary.csv}"
PROJECT_PREFIX="${MIDI_BRAINAGE_PROJECT_PREFIX:-midi_simon_strat12}"
CASE_TIMEOUT_SECONDS="${MIDI_BRAINAGE_CASE_TIMEOUT_SECONDS:-1800}"
LIMIT="${MIDI_BRAINAGE_LIMIT:-0}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Missing MIDIBrainAge Python executable: $VENV_PYTHON" >&2
  exit 2
fi

if [[ ! -f "$WORK_DIR/run_inference.py" ]]; then
  echo "Missing MIDIBrainAge run_inference.py in work dir: $WORK_DIR" >&2
  echo "Prepare the isolated MIDIBrainAge work directory before running this wrapper." >&2
  exit 2
fi

mkdir -p "$DATA_DIR"

"$VENV_PYTHON" "$EXP_DIR/run_midi_brainage_batch.py" \
  --manifest "$MANIFEST" \
  --output-csv "$OUTPUT_CSV" \
  --work-dir "$WORK_DIR" \
  --python "$VENV_PYTHON" \
  --project-prefix "$PROJECT_PREFIX" \
  --case-timeout-seconds "$CASE_TIMEOUT_SECONDS" \
  --limit "$LIMIT" \
  --return-metrics \
  --resume

"$VENV_PYTHON" "$EXP_DIR/summarize_midi_brainage_results.py" \
  --predictions-csv "$OUTPUT_CSV" \
  --output-csv "$SUMMARY_CSV" \
  --summary-id "midi_simon_stratified12" \
  --group-cols dataset branch \
  --claim-level "small_labeled_sanity_not_validation" \
  --interpretation "SIMON stratified-12 labeled sanity gate on existing FastSurfer orig.mgz derivatives; not a Kate biological-age validation claim."

echo "MIDIBrainAge SIMON predictions: $OUTPUT_CSV"
echo "MIDIBrainAge SIMON summary: $SUMMARY_CSV"
