#!/usr/bin/env python3
"""Build MIDIBrainAge manifests for local SIMON FastSurfer derivatives."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_SIMON_FASTSURFER_ROOT = Path("/mnt/d/data/fastserfer_simon")
DEFAULT_SIMON_PHENO = Path("/mnt/d/data/SIMON_pheno.csv")

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


def read_age_map(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        return {f"{int(row['Session']):03d}": row["Age"] for row in rows}


def parse_simon_orig(path: Path, age_by_session: dict[str, str]) -> dict[str, str]:
    name = path.name
    session_match = re.search(r"ses-(\d{3})", name)
    run_match = re.search(r"(run-\d+)", name)
    acq_match = re.search(r"(acq-[^_]+)", name)
    session = session_match.group(1) if session_match else ""
    scan_id = name.removesuffix("_orig.mgz")
    return {
        "dataset": "SIMON",
        "subject_id": "SIMON",
        "session": session,
        "run": run_match.group(1) if run_match else "",
        "acquisition": acq_match.group(1) if acq_match else "",
        "scan_id": scan_id,
        "branch": "midi_simon_fastsurfer_orig",
        "preprocessing_level": "existing_fastsurfer_orig_mgz_conformed",
        "chronological_age_years": age_by_session.get(session, ""),
        "path": str(path),
        "source_archive": "",
        "notes": "Existing FastSurfer orig.mgz derivative; not raw SIMON BIDS/NIfTI.",
    }


def load_all_rows(simon_root: Path, simon_pheno: Path) -> list[dict[str, str]]:
    age_by_session = read_age_map(simon_pheno)
    return [parse_simon_orig(path, age_by_session) for path in sorted(simon_root.glob("*_orig.mgz"))]


def session_first(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["session"], []).append(row)

    chosen = []
    for session in sorted(grouped):
        candidates = sorted(grouped[session], key=lambda row: row["scan_id"])

        def score(row: dict[str, str]) -> tuple[int, str]:
            if row["run"] == "run-1" and row["acquisition"] == "":
                return (0, row["scan_id"])
            if row["run"] == "" and row["acquisition"] == "":
                return (1, row["scan_id"])
            if row["run"] == "run-1":
                return (2, row["scan_id"])
            return (3, row["scan_id"])

        chosen.append(sorted(candidates, key=score)[0])
    return chosen


def stratified(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or count >= len(rows):
        return rows
    ordered = sorted(rows, key=lambda row: float(row["chronological_age_years"]))
    if count == 1:
        return [ordered[0]]
    indices = sorted({round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)})
    return [ordered[i] for i in indices]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simon-fastsurfer-root", type=Path, default=DEFAULT_SIMON_FASTSURFER_ROOT)
    parser.add_argument("--simon-pheno", type=Path, default=DEFAULT_SIMON_PHENO)
    parser.add_argument("--mode", choices=["all-orig", "session-first", "stratified"], default="session-first")
    parser.add_argument("--stratified-count", type=int, default=12)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    rows = load_all_rows(args.simon_fastsurfer_root, args.simon_pheno)
    if args.mode in {"session-first", "stratified"}:
        rows = session_first(rows)
    if args.mode == "stratified":
        rows = stratified(rows, args.stratified_count)
    write_rows(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
