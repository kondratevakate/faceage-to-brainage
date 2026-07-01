#!/usr/bin/env bash
set -euo pipefail

# Reproducible wrapper for the SIMON all-orig MIDIBrainAge branch.
#
# Workflow captured here:
# - input manifest: FastSurfer/FreeSurfer-style `_orig.mgz` internal source
#   images listed in experiments/kate_n1_2026/midi_brainage_simon_all_orig_inputs.csv;
# - execution environment: WSL Ubuntu, isolated MIDIBrainAge Python 3.11 venv;
# - tools: MIDIBrainAge run_inference.py, HD-BET/ANTs as called by MIDIBrainAge,
#   run_midi_brainage_batch.py for resumable batch execution, and
#   summarize_midi_brainage_results.py for branch-level/session-level summaries;
# - interpretation guard: outputs characterize this input/preprocessing branch
#   on SIMON and are not a Kate biological-age claim.

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/kate_n1_2026"

VENV_PYTHON="${MIDI_BRAINAGE_PYTHON:-/home/kate/.venvs/midi_brainage_py311/bin/python}"
WORK_DIR="${MIDI_BRAINAGE_WORK_DIR:-/home/kate/midi_brainage_work}"
MANIFEST="${MIDI_BRAINAGE_MANIFEST:-$EXP_DIR/midi_brainage_simon_all_orig_inputs.csv}"
OUTPUT_CSV="${MIDI_BRAINAGE_OUTPUT_CSV:-$DATA_DIR/midi_brainage_simon_all_orig_predictions.csv}"
SUMMARY_CSV="${MIDI_BRAINAGE_SUMMARY_CSV:-$DATA_DIR/midi_brainage_simon_all_orig_summary.csv}"
SESSION_SUMMARY_CSV="${MIDI_BRAINAGE_SESSION_SUMMARY_CSV:-$DATA_DIR/midi_brainage_simon_all_orig_session_summary.csv}"
SEED_CSV="${MIDI_BRAINAGE_SEED_CSV:-$DATA_DIR/midi_brainage_simon_stratified12_predictions.csv}"
PROJECT_PREFIX="${MIDI_BRAINAGE_PROJECT_PREFIX:-midi_simon_allorig}"
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

if [[ ! -f "$OUTPUT_CSV" && -f "$SEED_CSV" ]]; then
  "$VENV_PYTHON" - "$MANIFEST" "$SEED_CSV" "$OUTPUT_CSV" <<'PY'
import csv
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
seed_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])

with manifest_path.open(newline="", encoding="utf-8") as handle:
    manifest_rows = list(csv.DictReader(handle))
manifest_scan_ids = {row["scan_id"] for row in manifest_rows}

with seed_path.open(newline="", encoding="utf-8") as handle:
    seed_reader = csv.DictReader(handle)
    seed_rows = [
        row
        for row in seed_reader
        if row.get("scan_id") in manifest_scan_ids and row.get("status") == "ok"
    ]
    fieldnames = seed_reader.fieldnames or []

if seed_rows:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seed_rows)
    print(f"Seeded {len(seed_rows)} completed predictions from {seed_path} into {output_path}")
PY
fi

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
  --summary-id "midi_simon_all_orig" \
  --group-cols dataset branch \
  --claim-level "branch_characterization_not_kate_claim" \
  --interpretation "SIMON all-orig MIDIBrainAge run on existing FastSurfer orig.mgz internal source images; repeated/acquisition variants are not independent subjects and this is not a Kate biological-age validation claim."

"$VENV_PYTHON" "$EXP_DIR/summarize_midi_brainage_results.py" \
  --predictions-csv "$OUTPUT_CSV" \
  --output-csv "$SESSION_SUMMARY_CSV" \
  --summary-id "midi_simon_all_orig_by_session" \
  --group-cols session \
  --claim-level "repeat_acquisition_qc_not_kate_claim" \
  --interpretation "Per-session summary for SIMON all-orig FastSurfer orig.mgz internal source images; multi-run/acquisition rows estimate branch sensitivity, not independent validation."

echo "MIDIBrainAge SIMON all-orig predictions: $OUTPUT_CSV"
echo "MIDIBrainAge SIMON all-orig summary: $SUMMARY_CSV"
echo "MIDIBrainAge SIMON all-orig session summary: $SESSION_SUMMARY_CSV"
