#!/usr/bin/env python3
"""Build local BrainIAC brain-age input manifests for Kate and SIMON."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


KATE_ROOT = Path("/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years")
SIMON_FASTSURFER_ROOT = Path("/mnt/d/data/fastserfer_simon")
SIMON_PHENO = Path("/mnt/d/data/SIMON_pheno.csv")


KATE_RAW_INPUTS = [
    ("kate_2018_t1_ge_fspgr", "2018", "raw_direct_t1like", "images/2018/nifti/3_fspgr_bravo_10mm_ax.nii.gz"),
    ("kate_2022_t1_se_sag", "2022", "raw_direct_t1like", "images/2022/nifti/4_t1_se_sag.nii.gz"),
    ("kate_2024_t1_ffe_401", "2024", "raw_direct_t1like", "images/2024/nifti/401_t1w_ffe.nii.gz"),
    ("kate_2024_t1_ffe_601", "2024", "raw_direct_t1like", "images/2024/nifti/601_t1w_ffe.nii.gz"),
    ("kate_2024_3di_901", "2024", "raw_direct_t1like", "images/2024/nifti/901_3di_mc_hr.nii.gz"),
]

KATE_TIGERBX_TBET_INPUTS = [
    (
        "kate_2018_t1_ge_fspgr_tigerbx_tbet",
        "2018",
        "existing_tigerbx_tbet",
        "reprocessed_2026/asian_mri_tools/tigerbx/bx/kate_2018_t1_tbet.nii.gz",
    ),
    (
        "kate_2022_t1_se_sag_tigerbx_tbet",
        "2022",
        "existing_tigerbx_tbet",
        "reprocessed_2026/asian_mri_tools/tigerbx/bx/kate_2022_t1_tbet.nii.gz",
    ),
    (
        "kate_2024_t1_ffe_401_tigerbx_tbet",
        "2024",
        "existing_tigerbx_tbet",
        "reprocessed_2026/asian_mri_tools/tigerbx/bx/kate_2024_t1_ffe_401_tbet.nii.gz",
    ),
    (
        "kate_2024_t1_ffe_601_tigerbx_tbet",
        "2024",
        "existing_tigerbx_tbet",
        "reprocessed_2026/asian_mri_tools/tigerbx/bx/kate_2024_t1_ffe_601_tbet.nii.gz",
    ),
    (
        "kate_2024_3di_901_tigerbx_tbet",
        "2024",
        "existing_tigerbx_tbet",
        "reprocessed_2026/asian_mri_tools/tigerbx/bx/kate_2024_3di_tbet.nii.gz",
    ),
]


FIELDNAMES = [
    "dataset",
    "subject_id",
    "session",
    "run",
    "scan_id",
    "branch",
    "preprocessing_level",
    "chronological_age_years",
    "path",
    "notes",
]


def read_simon_age_map(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {f"{int(row['Session']):03d}": row["Age"] for row in rows}


def add_kate_rows(rows: list[dict[str, str]], kate_root: Path) -> None:
    for scan_id, session, branch, rel_path in KATE_RAW_INPUTS:
        rows.append(
            {
                "dataset": "kate_n1_2026",
                "subject_id": "kate",
                "session": session,
                "run": "",
                "scan_id": scan_id,
                "branch": branch,
                "preprocessing_level": "none_except_model_resize_scale",
                "chronological_age_years": "",
                "path": str(kate_root / rel_path),
                "notes": "Direct local T1-like NIfTI input; no BrainIAC optional preprocessing.",
            }
        )

    for scan_id, session, branch, rel_path in KATE_TIGERBX_TBET_INPUTS:
        rows.append(
            {
                "dataset": "kate_n1_2026",
                "subject_id": "kate",
                "session": session,
                "run": "",
                "scan_id": scan_id,
                "branch": branch,
                "preprocessing_level": "existing_tigerbx_brain_extracted_not_mni_registered",
                "chronological_age_years": "",
                "path": str(kate_root / rel_path),
                "notes": "Existing TIGERBx tBET output; not a fresh BrainIAC full-preprocessing run.",
            }
        )


def add_simon_fastsurfer_rows(rows: list[dict[str, str]], simon_root: Path, simon_pheno: Path) -> None:
    age_by_session = read_simon_age_map(simon_pheno)
    for path in sorted(simon_root.glob("*_orig.mgz")):
        match = re.search(r"ses-(\d{3})", path.name)
        session = match.group(1) if match else ""
        stem = path.name.removesuffix("_orig.mgz")
        run_match = re.search(r"(run-\d+)", stem)
        rows.append(
            {
                "dataset": "SIMON",
                "subject_id": "SIMON",
                "session": session,
                "run": run_match.group(1) if run_match else "",
                "scan_id": stem,
                "branch": "simon_fastsurfer_orig_conformed",
                "preprocessing_level": "existing_fastsurfer_orig_mgz_conformed",
                "chronological_age_years": age_by_session.get(session, ""),
                "path": str(path),
                "notes": "Existing FastSurfer orig.mgz derivative; local raw SIMON NIfTI/BIDS source not found.",
            }
        )


def write_blockers(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "branch", "status", "blocker", "checked_paths", "next_action"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "SIMON",
                "branch": "raw_direct_no_preprocessing",
                "status": "blocked",
                "blocker": "No local SIMON raw T1 NIfTI/BIDS source was found; /mnt/mydisk is not mounted and D:/data contains derivatives.",
                "checked_paths": "/mnt/mydisk; /mnt/d/data; /mnt/d/data/fastserfer_simon; /mnt/d/data/freesurfer8_simon",
                "next_action": "Mount or provide raw SIMON sourcedata, then add a direct raw manifest branch.",
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kate-root", type=Path, default=KATE_ROOT)
    parser.add_argument("--simon-fastsurfer-root", type=Path, default=SIMON_FASTSURFER_ROOT)
    parser.add_argument("--simon-pheno", type=Path, default=SIMON_PHENO)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--blockers-csv", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    add_kate_rows(rows, args.kate_root)
    add_simon_fastsurfer_rows(rows, args.simon_fastsurfer_root, args.simon_pheno)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    write_blockers(args.blockers_csv)

    print(f"Wrote {len(rows)} rows: {args.output_manifest}")
    print(f"Wrote blocker notes: {args.blockers_csv}")


if __name__ == "__main__":
    main()
