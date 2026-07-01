#!/usr/bin/env python3
"""Build MIDIBrainAge manifests for Kate n=1 T1-like local inputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years")
DEFAULT_FOUNDATION_INPUTS = Path("experiments/kate_n1_2026/foundation_model_inputs.csv")

FIELDNAMES = [
    "dataset",
    "subject_id",
    "session",
    "run",
    "acquisition",
    "scan_id",
    "branch",
    "preprocessing_level",
    "chronological_age_years",
    "path",
    "source_archive",
    "notes",
]


def load_rows(foundation_inputs: Path, data_root: Path) -> list[dict[str, str]]:
    rows = []
    with foundation_inputs.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("brainiac_candidate") != "1":
                continue
            path = data_root / row["relative_path"]
            rows.append(
                {
                    "dataset": "kate_n1_2026",
                    "subject_id": "kate",
                    "session": row["session"],
                    "run": "",
                    "acquisition": row["modality_hint"],
                    "scan_id": row["scan_id"],
                    "branch": "midi_kate_t1_like_raw_nifti",
                    "preprocessing_level": "raw_nifti_midibrainage_internal_skullstrip_register",
                    "chronological_age_years": "",
                    "path": str(path),
                    "source_archive": "",
                    "notes": row["notes"],
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--foundation-inputs", type=Path, default=DEFAULT_FOUNDATION_INPUTS)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    rows = load_rows(args.foundation_inputs, args.data_root)
    write_rows(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
