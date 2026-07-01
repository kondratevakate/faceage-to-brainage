#!/usr/bin/env python3
"""Extract per-label volumes from NIfTI label maps."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_scan_and_type(path: Path) -> tuple[str, str]:
    name = path.name.removesuffix(".nii.gz").removesuffix(".nii")
    suffixes = [
        "_aseg",
        "_dgm",
        "_syn",
        "_hlc",
        "_tbetmask",
        "_wmh",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix[1:]
    return name, "labelmap"


def load_lut(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {int(row["label"]): row.get("label_name", "") for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--lut", type=Path)
    args = parser.parse_args()

    lut = load_lut(args.lut)
    files: list[Path] = []
    for pattern in args.input_glob:
        files.extend(Path(p) for p in glob.glob(pattern))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No label maps matched: {args.input_glob}")

    rows = []
    for path in files:
        scan_id, output_type = infer_scan_and_type(path)
        img = nib.load(str(path))
        data = np.rint(np.asarray(img.get_fdata(dtype=np.float32))).astype(np.int64)
        labels, counts = np.unique(data, return_counts=True)
        voxel_volume_mm3 = float(abs(np.linalg.det(img.affine[:3, :3])))
        for label, count in zip(labels, counts):
            if int(label) == 0:
                continue
            volume_mm3 = float(count) * voxel_volume_mm3
            rows.append(
                {
                    "method": args.method,
                    "scan_id": scan_id,
                    "output_type": output_type,
                    "label": int(label),
                    "label_name": lut.get(int(label), ""),
                    "voxels": int(count),
                    "voxel_volume_mm3": voxel_volume_mm3,
                    "volume_mm3": volume_mm3,
                    "volume_ml": volume_mm3 / 1000.0,
                    "labelmap_path": str(path),
                    "labelmap_sha256": sha256_file(path),
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method",
            "scan_id",
            "output_type",
            "label",
            "label_name",
            "voxels",
            "voxel_volume_mm3",
            "volume_mm3",
            "volume_ml",
            "labelmap_path",
            "labelmap_sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} volume rows to {args.output_csv}")


if __name__ == "__main__":
    main()
