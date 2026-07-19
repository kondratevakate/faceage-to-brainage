#!/usr/bin/env python3
"""Build the locked BNU1 T1 inventory and paired repeatability inclusion table."""

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
    "/mnt/d/data/faceage-to-brainage/sourcedata/bnu1_corr/"
    "s3_2026-07-18/BNU_1"
)
DEFAULT_OUTPUT = Path("data/brainage/bnu1_t1w_inclusion.csv")
DEFAULT_METADATA = Path("data/brainage/bnu1_t1w_inclusion_metadata.json")
SCAN_PATTERN = re.compile(
    r"^(sub-\d+)/ses-([12])/anat/(sub-\d+)_ses-([12])_run-1_T1w\.nii\.gz$"
)
EXPECTED_PARTICIPANTS = 57
EXPECTED_T1_SCANS = 107
EXPECTED_SOURCE_PAIRS = 50
EXPECTED_ANALYSIS_PAIRS = 49
EXPECTED_SHAPE = (144, 256, 256)
EXPECTED_HEADER_QC_FAILURES = {
    "sub-0025913/ses-2/anat/sub-0025913_ses-2_run-1_T1w.nii.gz"
}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_demographics(dataset_root: Path) -> dict[str, dict[str, object]]:
    demographics: dict[str, dict[str, object]] = {}
    for sessions_path in sorted(dataset_root.glob("sub-*/sub-*_sessions.tsv")):
        rows = read_csv(sessions_path, delimiter="\t")
        if len(rows) != 2 or {row["session_id"] for row in rows} != {"ses-1", "ses-2"}:
            raise ValueError(f"Unexpected session table: {sessions_path}")
        session_1 = next(row for row in rows if row["session_id"] == "ses-1")
        session_2 = next(row for row in rows if row["session_id"] == "ses-2")
        participant = f"sub-{session_1['participant_id']}"
        if sessions_path.parent.name != participant:
            raise ValueError(f"Participant ID/path mismatch: {sessions_path}")
        if participant in demographics:
            raise ValueError(f"Duplicate session table: {participant}")
        age = float(session_1["age_at_scan_1"])
        duration_days = int(session_2["retest_duration"])
        if not math.isfinite(age) or age <= 0 or duration_days <= 0:
            raise ValueError(f"Invalid age or retest duration: {sessions_path}")
        demographics[participant] = {
            "reported_sex": session_1["sex"],
            "age_at_session_1_years": age,
            "retest_duration_days": duration_days,
            "sessions_tsv_relative_path": sessions_path.relative_to(
                dataset_root
            ).as_posix(),
            "sessions_tsv_sha256": sha256_file(sessions_path),
        }
    if len(demographics) != EXPECTED_PARTICIPANTS:
        raise ValueError(
            f"Expected {EXPECTED_PARTICIPANTS} session tables, found {len(demographics)}"
        )
    return demographics


def format_triplet(values: tuple[float, ...]) -> str:
    return "x".join(f"{float(value):.6g}" for value in values[:3])


def inspect_scan(
    scan_path: Path,
    dataset_root: Path,
    demographics: dict[str, dict[str, object]],
    provenance: dict[str, dict[str, str]],
    complete_participants: set[str],
) -> dict[str, str]:
    relative = scan_path.relative_to(dataset_root).as_posix()
    match = SCAN_PATTERN.fullmatch(relative)
    if match is None:
        raise ValueError(f"Unexpected T1 path: {relative}")
    participant, session, filename_participant, filename_session = match.groups()
    session_id = f"ses-{session}"
    if participant != filename_participant or session != filename_session:
        raise ValueError(f"Filename/path mismatch: {relative}")
    if participant not in demographics:
        raise ValueError(f"Missing demographics: {participant}")
    if relative not in provenance:
        raise ValueError(f"Missing acquisition provenance: {relative}")

    source_sha256 = sha256_file(scan_path)
    source_record = provenance[relative]
    if (
        source_sha256 != source_record["sha256"]
        or scan_path.stat().st_size != int(source_record["size_bytes"])
    ):
        raise ValueError(f"Source hash or size changed: {relative}")

    image = nib.load(str(scan_path))
    data = np.asanyarray(image.dataobj)
    total_voxels = int(np.prod(image.shape[:3]))
    finite = bool(np.isfinite(data).all())
    nonzero_voxels = int(np.count_nonzero(data))
    zooms = image.header.get_zooms()[:3]
    exclusion_reasons: list[str] = []
    if len(image.shape) != 3:
        exclusion_reasons.append("not_3d")
    if image.shape != EXPECTED_SHAPE:
        exclusion_reasons.append("unexpected_shape_vs_t1w_metadata")
    if not finite:
        exclusion_reasons.append("nonfinite_voxels")
    if nonzero_voxels == 0:
        exclusion_reasons.append("empty_volume")
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in zooms):
        exclusion_reasons.append("invalid_voxel_size")

    demographic = demographics[participant]
    elapsed_days = 0 if session_id == "ses-1" else int(
        demographic["retest_duration_days"]
    )
    age_session_1 = float(demographic["age_at_session_1_years"])
    estimated_age = age_session_1 + elapsed_days / 365.2425
    complete_pair = participant in complete_participants
    header_qc_pass = not exclusion_reasons

    return {
        "dataset_id": "bnu1_corr",
        "release": "FCP_INDI_S3_snapshot_2026-07-18",
        "participant_id": participant,
        "session_id": session_id,
        "age_at_session_1_years": f"{age_session_1:.6g}",
        "elapsed_from_session_1_days": str(elapsed_days),
        "estimated_chronological_age_years": f"{estimated_age:.8g}",
        "chronological_age_precision": "integer_year_at_session_1_plus_exact_retest_days",
        "reported_sex": str(demographic["reported_sex"]),
        "relative_path": relative,
        "source_sha256": source_sha256,
        "source_s3_etag_md5": source_record["s3_etag_md5"],
        "source_size_bytes": str(scan_path.stat().st_size),
        "shape": "x".join(str(value) for value in image.shape),
        "voxel_size_mm": format_triplet(zooms),
        "orientation": "".join(nib.aff2axcodes(image.affine)),
        "dtype": str(image.get_data_dtype()),
        "nonzero_fraction": f"{nonzero_voxels / total_voxels:.8g}",
        "qform_code": str(int(image.header["qform_code"])),
        "sform_code": str(int(image.header["sform_code"])),
        "source_preprocessing": "FCP_INDI_deidentified_face_removed_per_CoRR_policy",
        "neurofm_age_range_status": "below_documented_40_to_90_range",
        "source_pair_complete": "1" if complete_pair else "0",
        "pair_role": "pending_pair_qc",
        "header_numeric_qc_include": "1" if header_qc_pass else "0",
        "paired_repeatability_include": "0",
        "header_exclusion_reason": ";".join(exclusion_reasons),
        "paired_exclusion_reason": "pending_pair_qc",
        "qc_state": "header_and_numeric_qc_pass" if header_qc_pass else "failed",
        "analysis_role": "test_retest_robustness_only_not_age_accuracy",
    }


def assign_pair_inclusion(
    rows: list[dict[str, str]], complete_participants: set[str]
) -> None:
    rows_by_participant: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_participant.setdefault(row["participant_id"], []).append(row)
    for participant, participant_rows in rows_by_participant.items():
        if participant not in complete_participants:
            for row in participant_rows:
                row["pair_role"] = "baseline_only_missing_session_2_t1"
                row["paired_exclusion_reason"] = "source_session_2_t1_missing"
            continue
        failed = [row for row in participant_rows if row["header_numeric_qc_include"] != "1"]
        if failed:
            for row in participant_rows:
                row["pair_role"] = "source_pair_qc_excluded"
                row["paired_exclusion_reason"] = (
                    "self_header_qc_failed"
                    if row["header_numeric_qc_include"] != "1"
                    else "counterpart_header_qc_failed"
                )
            continue
        for row in participant_rows:
            row["pair_role"] = "primary_complete_pair"
            row["paired_repeatability_include"] = "1"
            row["paired_exclusion_reason"] = ""


def validate_design(rows: list[dict[str, str]]) -> None:
    if len(rows) != EXPECTED_T1_SCANS:
        raise ValueError(f"Expected {EXPECTED_T1_SCANS} T1 scans, found {len(rows)}")
    failed = {
        row["relative_path"]
        for row in rows
        if row["header_numeric_qc_include"] != "1"
    }
    if failed != EXPECTED_HEADER_QC_FAILURES:
        raise ValueError(f"Unexpected header/numeric QC failure set: {sorted(failed)}")
    session_counts = Counter(row["session_id"] for row in rows)
    if session_counts != Counter({"ses-1": 57, "ses-2": 50}):
        raise ValueError(f"Unexpected T1 session counts: {session_counts}")
    paired = [row for row in rows if row["paired_repeatability_include"] == "1"]
    if len(paired) != EXPECTED_ANALYSIS_PAIRS * 2:
        raise ValueError(f"Expected 98 primary paired scan rows, found {len(paired)}")
    paired_counts = Counter(row["participant_id"] for row in paired)
    if len(paired_counts) != EXPECTED_ANALYSIS_PAIRS or set(paired_counts.values()) != {2}:
        raise ValueError("Paired cohort is not a complete 49 x 2 design")
    keys = [(row["participant_id"], row["session_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate participant/session key")


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

    provenance_path = args.dataset_root / "_provenance" / "download_manifest.csv"
    provenance_rows = read_csv(provenance_path)
    provenance = {row["relative_path"]: row for row in provenance_rows}
    demographics = read_demographics(args.dataset_root)
    scans = sorted(args.dataset_root.glob("sub-*/ses-*/anat/*_T1w.nii.gz"))
    available_sessions: dict[str, set[str]] = {}
    for scan_path in scans:
        relative_parts = scan_path.relative_to(args.dataset_root).parts
        available_sessions.setdefault(relative_parts[0], set()).add(relative_parts[1])
    complete_participants = {
        participant
        for participant, sessions in available_sessions.items()
        if sessions == {"ses-1", "ses-2"}
    }
    rows = [
        inspect_scan(
            scan_path,
            args.dataset_root,
            demographics,
            provenance,
            complete_participants,
        )
        for scan_path in scans
    ]
    rows.sort(key=lambda row: (row["participant_id"], row["session_id"]))
    assign_pair_inclusion(rows, complete_participants)
    validate_design(rows)
    write_csv(args.output, rows)

    ages = [float(value["age_at_session_1_years"]) for value in demographics.values()]
    durations = [int(value["retest_duration_days"]) for value in demographics.values()]
    sex_counts = Counter(str(value["reported_sex"]) for value in demographics.values())
    metadata = {
        "dataset_id": "bnu1_corr",
        "release": "FCP_INDI_S3_snapshot_2026-07-18",
        "dataset_doi": "10.15387/fcp_indi.corr.bnu1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "download_manifest_sha256": sha256_file(provenance_path),
        "participants_tsv_sha256": sha256_file(args.dataset_root / "participants.tsv"),
        "t1w_json_sha256": sha256_file(args.dataset_root / "T1w.json"),
        "n_participants_metadata": len(demographics),
        "n_t1_scans": len(rows),
        "n_session_1_t1": sum(row["session_id"] == "ses-1" for row in rows),
        "n_session_2_t1": sum(row["session_id"] == "ses-2" for row in rows),
        "n_complete_source_t1_pairs": len(complete_participants),
        "n_primary_qc_passed_t1_pairs": EXPECTED_ANALYSIS_PAIRS,
        "n_header_numeric_qc_pass": sum(
            row["header_numeric_qc_include"] == "1" for row in rows
        ),
        "n_header_numeric_qc_fail": sum(
            row["header_numeric_qc_include"] != "1" for row in rows
        ),
        "header_numeric_qc_failures": sorted(EXPECTED_HEADER_QC_FAILURES),
        "paired_qc_excluded_participants": ["sub-0025913"],
        "n_baseline_only": EXPECTED_PARTICIPANTS - len(complete_participants),
        "baseline_only_participants": sorted(set(demographics) - complete_participants),
        "baseline_age_years": {
            "minimum": min(ages),
            "maximum": max(ages),
            "mean": float(np.mean(ages)),
        },
        "reported_sex_counts": dict(sorted(sex_counts.items())),
        "retest_duration_days": {
            "minimum": min(durations),
            "maximum": max(durations),
            "mean": float(np.mean(durations)),
            "sd": float(np.std(durations, ddof=1)),
        },
        "documented_neurofm_age_range_years": [40, 90],
        "all_participants_below_neurofm_documented_age_range": True,
        "source_face_status": "face_removed_per_CoRR_data_aggregation_policy",
        "permitted_claim": "test_retest_robustness_only_not_age_accuracy",
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Locked BNU1 inventory: {args.output} "
        f"({len(rows)} T1 scans, {EXPECTED_ANALYSIS_PAIRS} primary pairs)"
    )


if __name__ == "__main__":
    main()
