#!/usr/bin/env python3
"""Convert existing SIMON FastSurfer orig.mgz images to NIfTI for NeuroFM.

This prepares an explicit raw-orig diagnostic branch. It does not skull-strip,
conform, or otherwise claim to match NeuroFM's recommended preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def row_image_path(row: dict[str, str]) -> Path:
    for key in ("image_path", "path", "input", "input_path_wsl"):
        value = row.get(key, "")
        if value:
            return Path(value)
    raise KeyError("Input row must contain one of image_path/path/input/input_path_wsl")


def shape_text(values: tuple[int, ...]) -> str:
    return "x".join(str(v) for v in values)


def zoom_text(img: nib.spatialimages.SpatialImage) -> str:
    return "x".join(f"{float(v):.6g}" for v in img.header.get_zooms()[:3])


def safe_stem(scan_id: str, index: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in scan_id)
    return safe or f"scan_{index:04d}"


def process_row(row: dict[str, str], index: int, output_dir: Path, overwrite: bool) -> dict[str, str]:
    out = dict(row)
    image_path = row_image_path(row)
    scan_id = row.get("scan_id", "") or image_path.name.removesuffix(".mgz")
    output_path = output_dir / f"{safe_stem(scan_id, index)}_neurofm_raw_orig.nii.gz"
    out.update(
        {
            "source_image": str(image_path),
            "input": str(output_path),
            "status": "failed",
            "error": "",
            "image_shape": "",
            "image_zooms_mm": "",
            "image_dtype": "",
            "image_sha256": "",
            "converted_input_sha256": "",
            "preprocessing_detail": "mgz_to_nii_gz_no_skullstrip",
        }
    )
    try:
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        img = nib.load(str(image_path))
        out["image_shape"] = shape_text(img.shape[:3])
        out["image_zooms_mm"] = zoom_text(img)
        data = np.asanyarray(img.dataobj)
        out["image_dtype"] = str(data.dtype)
        out["image_sha256"] = sha256_file(image_path)

        if not output_path.exists() or overwrite:
            output_dir.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(data, img.affine), str(output_path))
        out["converted_input_sha256"] = sha256_file(output_path)
        out["status"] = "ok"
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--status-csv", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.input_manifest)
    if args.limit > 0:
        rows = rows[: args.limit]

    status_rows = [process_row(row, index, args.output_dir, args.overwrite) for index, row in enumerate(rows, start=1)]
    ok_rows = [row for row in status_rows if row["status"] == "ok"]
    write_csv(args.status_csv, status_rows)
    write_csv(args.output_manifest, ok_rows, fieldnames=list(status_rows[0].keys()) if status_rows else ["input"])
    print(f"Prepared {len(ok_rows)}/{len(status_rows)} NeuroFM raw-orig NIfTI inputs")


if __name__ == "__main__":
    main()
