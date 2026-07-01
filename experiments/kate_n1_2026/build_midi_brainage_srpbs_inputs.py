#!/usr/bin/env python3
"""Build MIDIBrainAge manifests for SRPBS Traveling Subjects inputs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_FASTSURFER_ROOT = Path("/mnt/d/data/fastserfer_travelling")
DEFAULT_RAW_ROOT = Path("/home/kate/srpbs_ts_raw_t1w/SRPBS_TS/sourcedata")
DEFAULT_PARTICIPANTS_TSV = Path("data/kate_n1_2026/srpbs_travelling_participants.tsv")

FIELDNAMES = [
    "dataset",
    "subject_id",
    "session",
    "site",
    "run",
    "scan_id",
    "branch",
    "preprocessing_level",
    "chronological_age_years",
    "path",
    "source_archive",
    "notes",
]


def parse_scan_id(scan_id: str) -> tuple[str, str, str]:
    subject_match = re.search(r"(sub-\d+)", scan_id)
    session_match = re.search(r"(ses-[^_]+)", scan_id)
    run_match = re.search(r"(run-\d+)", scan_id)
    subject = subject_match.group(1) if subject_match else ""
    session = session_match.group(1) if session_match else ""
    run = run_match.group(1) if run_match else ""
    return subject, session, run


def read_age_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return {row["participant_id"]: row["age"] for row in rows if row.get("participant_id")}


def row_from_path(
    path: Path,
    scan_id: str,
    branch: str,
    preprocessing_level: str,
    source_archive: str,
    age_by_subject: dict[str, str],
) -> dict[str, str]:
    subject, session, run = parse_scan_id(scan_id)
    return {
        "dataset": "SRPBS_TS",
        "subject_id": subject,
        "session": session,
        "site": session.removeprefix("ses-"),
        "run": run,
        "scan_id": scan_id,
        "branch": branch,
        "preprocessing_level": preprocessing_level,
        "chronological_age_years": age_by_subject.get(subject, ""),
        "path": str(path),
        "source_archive": source_archive,
        "notes": "Traveling-subject dataset; age labels support small sanity checks, while the full value is site/test-retest robustness.",
    }


def fastsurfer_rows(root: Path, age_by_subject: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    for path in sorted(root.glob("*_orig.mgz")):
        scan_id = path.name.removesuffix("_orig.mgz")
        rows.append(
            row_from_path(
                path=path,
                scan_id=scan_id,
                branch="midi_srpbs_travelling_fastsurfer_orig",
                preprocessing_level="existing_fastsurfer_orig_mgz_conformed",
                source_archive="",
                age_by_subject=age_by_subject,
            )
        )
    return rows


def raw_t1w_rows(root: Path, age_by_subject: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    for path in sorted(root.glob("sub-*/ses-*/anat/*_T1w.nii.gz")):
        scan_id = path.name.removesuffix(".nii.gz")
        rows.append(
            row_from_path(
                path=path,
                scan_id=scan_id,
                branch="midi_srpbs_travelling_raw_t1w",
                preprocessing_level="raw_bids_t1w_midibrainage_internal_skullstrip_register",
                source_archive="/mnt/d/data/SRPBS_TS.tar.gz",
                age_by_subject=age_by_subject,
            )
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def apply_filters(
    rows: list[dict[str, str]],
    subjects: list[str],
    sites: list[str],
    limit: int,
) -> list[dict[str, str]]:
    if subjects:
        keep_subjects = set(subjects)
        rows = [row for row in rows if row["subject_id"] in keep_subjects]
    if sites:
        keep_sites = set(sites)
        rows = [row for row in rows if row["site"] in keep_sites]
    if limit > 0:
        rows = rows[:limit]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["fastsurfer-orig", "raw-t1w"], default="fastsurfer-orig")
    parser.add_argument("--fastsurfer-root", type=Path, default=DEFAULT_FASTSURFER_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--participants-tsv", type=Path, default=DEFAULT_PARTICIPANTS_TSV)
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--site", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    age_by_subject = read_age_map(args.participants_tsv)

    if args.source == "fastsurfer-orig":
        rows = fastsurfer_rows(args.fastsurfer_root, age_by_subject=age_by_subject)
    else:
        rows = raw_t1w_rows(args.raw_root, age_by_subject=age_by_subject)

    rows = apply_filters(rows, subjects=args.subject, sites=args.site, limit=args.limit)
    write_rows(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
