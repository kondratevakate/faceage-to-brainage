#!/usr/bin/env python3
"""Build and validate the locked Maclaren ds000239 T1w inclusion manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_DATASET_ROOT = Path(
    "/mnt/d/data/faceage-to-brainage/sourcedata/maclaren_ds000239/"
    "R1.0.1/ds000239_R1.0.1"
)
DEFAULT_OUTPUT = Path("data/brainage/maclaren_t1w_inclusion.csv")
DEFAULT_METADATA = Path("data/brainage/maclaren_t1w_inclusion_metadata.json")
SCAN_PATTERN = re.compile(r"^(sub-\d+)_run-(\d+)_T1w\.nii\.gz$")
EXPECTED_SUBJECTS = {"sub-01", "sub-02", "sub-03"}
EXPECTED_RUNS = set(range(1, 41))


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_participants(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    participants = {row["participant_id"]: row for row in rows}
    if set(participants) != EXPECTED_SUBJECTS:
        raise ValueError(f"Unexpected participant IDs: {sorted(participants)}")
    return participants


def format_triplet(values: tuple[float, ...]) -> str:
    return "x".join(f"{float(value):.6g}" for value in values[:3])


def inspect_scan(
    scan_path: Path,
    dataset_root: Path,
    participants: dict[str, dict[str, str]],
) -> dict[str, str]:
    match = SCAN_PATTERN.fullmatch(scan_path.name)
    if match is None:
        raise ValueError(f"Unexpected T1w filename: {scan_path.name}")

    participant_id = match.group(1)
    run_index = int(match.group(2))
    if scan_path.parent.parent.name != participant_id:
        raise ValueError(f"Filename/path participant mismatch: {scan_path}")
    if participant_id not in participants:
        raise ValueError(f"Missing participant metadata: {participant_id}")

    image = nib.load(str(scan_path))
    data = np.asanyarray(image.dataobj)
    total_voxels = int(np.prod(image.shape[:3]))
    finite = bool(np.isfinite(data).all())
    nonzero_voxels = int(np.count_nonzero(data))
    age = float(participants[participant_id]["age"])

    exclusion_reasons: list[str] = []
    if len(image.shape) != 3:
        exclusion_reasons.append("not_3d")
    if not finite:
        exclusion_reasons.append("nonfinite_voxels")
    if nonzero_voxels == 0:
        exclusion_reasons.append("empty_volume")
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in image.header.get_zooms()[:3]):
        exclusion_reasons.append("invalid_voxel_size")

    return {
        "dataset_id": "maclaren_ds000239",
        "release": "R1.0.1",
        "participant_id": participant_id,
        "run_index": str(run_index),
        "chronological_age_years": f"{age:.6g}",
        "reported_sex": participants[participant_id]["gender"],
        "relative_path": scan_path.relative_to(dataset_root).as_posix(),
        "source_sha256": sha256_file(scan_path),
        "source_size_bytes": str(scan_path.stat().st_size),
        "shape": "x".join(str(v) for v in image.shape),
        "voxel_size_mm": format_triplet(image.header.get_zooms()),
        "orientation": "".join(nib.aff2axcodes(image.affine)),
        "dtype": str(image.get_data_dtype()),
        "nonzero_fraction": f"{nonzero_voxels / total_voxels:.8g}",
        "qform_code": str(int(image.header["qform_code"])),
        "sform_code": str(int(image.header["sform_code"])),
        "source_preprocessing": "SPM12_defaced_not_skull_stripped",
        "neurofm_age_range_status": "below_documented_40_to_90_range",
        "analysis_role": "short_interval_test_retest_robustness_only",
        "include": "1" if not exclusion_reasons else "0",
        "exclusion_reason": ";".join(exclusion_reasons),
        "qc_state": "header_and_numeric_qc_pass" if not exclusion_reasons else "failed",
    }


def validate_complete_design(rows: list[dict[str, str]]) -> None:
    if len(rows) != 120:
        raise ValueError(f"Expected 120 T1w scans, found {len(rows)}")
    if any(row["include"] != "1" for row in rows):
        failed = [row["relative_path"] for row in rows if row["include"] != "1"]
        raise ValueError(f"Manifest contains excluded scans: {failed}")

    counts = Counter(row["participant_id"] for row in rows)
    if counts != Counter({subject: 40 for subject in EXPECTED_SUBJECTS}):
        raise ValueError(f"Expected 40 scans per participant, found {dict(counts)}")

    for subject in EXPECTED_SUBJECTS:
        runs = {int(row["run_index"]) for row in rows if row["participant_id"] == subject}
        if runs != EXPECTED_RUNS:
            raise ValueError(f"Unexpected run set for {subject}: {sorted(runs)}")

    keys = [(row["participant_id"], row["run_index"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate participant/run key in manifest")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()

    participants_path = args.dataset_root / "participants.tsv"
    participants = read_participants(participants_path)
    scans = sorted(args.dataset_root.glob("sub-*/anat/*_T1w.nii.gz"))
    rows = [inspect_scan(path, args.dataset_root, participants) for path in scans]
    rows.sort(key=lambda row: (row["participant_id"], int(row["run_index"])))
    validate_complete_design(rows)
    write_csv(args.output, rows)

    metadata = {
        "dataset_id": "maclaren_ds000239",
        "release": "R1.0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root_runtime_parameter": "MACLAREN_DATASET_ROOT",
        "participants_tsv_sha256": sha256_file(participants_path),
        "archive_sha256": "c3a05f7b0dc39208438a3ee4d6a728493c32718dc38a351525b5dca3fef48cf1",
        "n_scans": len(rows),
        "n_participants": len(EXPECTED_SUBJECTS),
        "runs_per_participant": 40,
        "all_included": True,
        "source_is_defaced": True,
        "source_is_skull_stripped": False,
        "documented_neurofm_age_range_years": [40, 90],
        "observed_ages_years": sorted({float(row["chronological_age_years"]) for row in rows}),
        "permitted_claim": "test_retest_robustness_only_not_age_accuracy",
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Locked manifest: {args.output} ({len(rows)} scans, 3 participants x 40 runs)")


if __name__ == "__main__":
    main()
