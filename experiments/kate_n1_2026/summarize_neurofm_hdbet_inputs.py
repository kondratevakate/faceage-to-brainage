#!/usr/bin/env python3
"""Summarize Kate HD-BET skull-stripped inputs prepared for NeuroFM."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np


CASES = [
    (
        "2018",
        "kate_2018_t1",
        "T1w_FSPGR_BRAVO",
        "images/2018/nifti/3_fspgr_bravo_10mm_ax.nii.gz",
        "Primary 2018 structural T1-like input; NeuroFM docs require skull-stripped T1w and internally conform to 1mm/LIA.",
    ),
    (
        "2022",
        "kate_2022_t1",
        "T1w_SE_sagittal",
        "images/2022/nifti/4_t1_se_sag.nii.gz",
        "Primary 2022 T1 input; thick-slice protocol limits anatomical recovery.",
    ),
    (
        "2024",
        "kate_2024_3di",
        "3D_inversion_recovery_like",
        "images/2024/nifti/901_3di_mc_hr.nii.gz",
        "2024 3DI stress/probe input; keep as QC branch, not promoted without downstream checks.",
    ),
    (
        "2024",
        "kate_2024_t1_ffe_401",
        "T1w_FFE",
        "images/2024/nifti/401_t1w_ffe.nii.gz",
        "2024 T1 FFE axial candidate; compare against 3DI and FFE 601.",
    ),
    (
        "2024",
        "kate_2024_t1_ffe_601",
        "T1w_FFE",
        "images/2024/nifti/601_t1w_ffe.nii.gz",
        "2024 T1 FFE sagittal candidate; compare against 3DI and FFE 401.",
    ),
]


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def shape_text(values: tuple[int, ...]) -> str:
    return "x".join(str(v) for v in values)


def zoom_text(img: nib.spatialimages.SpatialImage) -> str:
    return "x".join(f"{float(v):.6g}" for v in img.header.get_zooms()[:3])


def summarize_case(data_root: Path, output_dir: Path, session: str, scan_id: str, modality: str, rel_path: str, notes: str) -> dict[str, str]:
    raw_path = data_root / rel_path
    output_path = output_dir / f"{scan_id}_hdbet.nii.gz"
    mask_path = output_dir / f"{scan_id}_hdbet_bet.nii.gz"
    row = {
        "method": "HD_BET",
        "scan_id": scan_id,
        "session": session,
        "modality_hint": modality,
        "input_path": str(raw_path),
        "output_path": str(output_path),
        "mask_path": str(mask_path),
        "status": "failed",
        "error": "",
        "hd_bet_version": "2.0.1",
        "device": "cpu",
        "disable_tta": "1",
        "input_shape": "",
        "output_shape": "",
        "mask_shape": "",
        "input_zooms_mm": "",
        "output_zooms_mm": "",
        "mask_zooms_mm": "",
        "output_nonzero_voxels": "",
        "mask_nonzero_voxels": "",
        "mask_fraction": "",
        "input_sha256": "",
        "output_sha256": "",
        "mask_sha256": "",
        "output_size_bytes": "",
        "mask_size_bytes": "",
        "claim_level": "input_preprocessing_only_not_age_claim",
        "notes": notes,
    }
    try:
        for path in (raw_path, output_path, mask_path):
            if not path.exists():
                raise FileNotFoundError(path)
        raw_img = nib.load(str(raw_path))
        out_img = nib.load(str(output_path))
        mask_img = nib.load(str(mask_path))
        out_data = out_img.get_fdata(dtype=np.float32)
        mask_data = mask_img.get_fdata(dtype=np.float32) > 0
        total_voxels = int(np.prod(mask_img.shape[:3]))
        mask_nonzero = int(mask_data.sum())
        output_nonzero = int(np.count_nonzero(out_data))
        if mask_nonzero == 0 or output_nonzero == 0:
            raise ValueError("empty output or mask")
        row.update(
            {
                "status": "ok",
                "input_shape": shape_text(raw_img.shape[:3]),
                "output_shape": shape_text(out_img.shape[:3]),
                "mask_shape": shape_text(mask_img.shape[:3]),
                "input_zooms_mm": zoom_text(raw_img),
                "output_zooms_mm": zoom_text(out_img),
                "mask_zooms_mm": zoom_text(mask_img),
                "output_nonzero_voxels": str(output_nonzero),
                "mask_nonzero_voxels": str(mask_nonzero),
                "mask_fraction": f"{mask_nonzero / total_voxels:.8g}",
                "input_sha256": sha256_file(raw_path),
                "output_sha256": sha256_file(output_path),
                "mask_sha256": sha256_file(mask_path),
                "output_size_bytes": str(output_path.stat().st_size),
                "mask_size_bytes": str(mask_path.stat().st_size),
            }
        )
    except Exception as exc:
        row["error"] = repr(exc)
    return row


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--status-csv", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    args = parser.parse_args()

    rows = [
        summarize_case(args.data_root, args.output_dir, session, scan_id, modality, rel_path, notes)
        for session, scan_id, modality, rel_path, notes in CASES
    ]
    write_csv(args.status_csv, rows)

    input_rows = [
        {
            "dataset": "kate_n1_2026",
            "subject_id": "kate",
            "session": row["session"],
            "scan_id": f"{row['scan_id']}_hdbet",
            "chronological_age_years": "",
            "input": row["output_path"],
            "mask_source": "HD_BET_2.0.1_cpu_disable_tta",
            "preprocessing_level": "raw_t1w_hdbet_skull_stripped",
            "evidence_role": "application_qc_only",
            "notes": row["notes"],
        }
        for row in rows
        if row["status"] == "ok"
    ]
    write_csv(args.input_manifest, input_rows)
    print(f"Kate HD-BET summary: {sum(row['status'] == 'ok' for row in rows)}/{len(rows)} ok")


if __name__ == "__main__":
    main()
