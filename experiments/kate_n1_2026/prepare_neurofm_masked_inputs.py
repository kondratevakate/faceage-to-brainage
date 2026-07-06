#!/usr/bin/env python3
"""Prepare skull-stripped NeuroFM inputs from an image plus an existing mask/label.

This does not validate the mask. It only creates an explicit, auditable
preprocessing branch for NeuroFM, whose upstream inference requires skull-stripped
T1w NIfTI inputs.
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


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_fastsurfer_label_path(image_path: Path) -> Path:
    name = image_path.name
    if name.endswith("_orig.mgz"):
        return image_path.with_name(name[: -len("_orig.mgz")] + "_aparcDKT+aseg.mgz")
    if name.endswith("orig.mgz"):
        return image_path.with_name(name[: -len("orig.mgz")] + "aparcDKT+aseg.mgz")
    raise ValueError(f"Cannot infer FastSurfer label from image name: {image_path}")


def shape_text(values: tuple[int, ...]) -> str:
    return "x".join(str(v) for v in values)


def zoom_text(img: nib.spatialimages.SpatialImage) -> str:
    return "x".join(f"{float(v):.6g}" for v in img.header.get_zooms()[:3])


def row_image_path(row: dict[str, str]) -> Path:
    for key in ("image_path", "path", "input", "input_path_wsl"):
        value = row.get(key, "")
        if value:
            return Path(value)
    raise KeyError("Input row must contain one of image_path/path/input/input_path_wsl")


def make_output_name(scan_id: str, index: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in scan_id)
    if not safe:
        safe = f"scan_{index:04d}"
    return f"{safe}_neurofm_skullstripped.nii.gz"


def process_row(
    row: dict[str, str],
    index: int,
    output_dir: Path,
    infer_label: bool,
    overwrite: bool,
) -> dict[str, str]:
    out = dict(row)
    image_path = row_image_path(row)
    label_value = row.get("label_path", "")
    label_path = Path(label_value) if label_value else None
    if label_path is None and infer_label:
        label_path = infer_fastsurfer_label_path(image_path)
    if label_path is None:
        raise ValueError(f"No label_path for row {index}: {row}")

    scan_id = row.get("scan_id", "") or image_path.name
    output_path = output_dir / make_output_name(scan_id, index)
    out.update(
        {
            "source_image": str(image_path),
            "source_label": str(label_path),
            "input": str(output_path),
            "status": "failed",
            "error": "",
            "image_shape": "",
            "label_shape": "",
            "image_zooms_mm": "",
            "label_zooms_mm": "",
            "mask_nonzero_voxels": "",
            "mask_fraction": "",
            "masked_nonzero_voxels": "",
            "image_sha256": "",
            "label_sha256": "",
            "masked_input_sha256": "",
            "preprocessing_detail": "image_times_binary_mask_label_gt_0",
        }
    )

    try:
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label/mask: {label_path}")

        img = nib.load(str(image_path))
        label_img = nib.load(str(label_path))
        if img.shape[:3] != label_img.shape[:3]:
            raise ValueError(f"Shape mismatch: image {img.shape[:3]} vs label {label_img.shape[:3]}")

        out["image_shape"] = shape_text(img.shape[:3])
        out["label_shape"] = shape_text(label_img.shape[:3])
        out["image_zooms_mm"] = zoom_text(img)
        out["label_zooms_mm"] = zoom_text(label_img)
        out["image_sha256"] = sha256_file(image_path)
        out["label_sha256"] = sha256_file(label_path)

        if output_path.exists() and not overwrite:
            masked = nib.load(str(output_path)).get_fdata(dtype=np.float32)
        else:
            data = img.get_fdata(dtype=np.float32)
            labels = label_img.get_fdata(dtype=np.float32)
            mask = labels > 0
            masked = np.where(mask, data, 0).astype(np.float32, copy=False)
            output_dir.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(masked, img.affine, img.header), str(output_path))

        mask_nonzero = int(np.count_nonzero(label_img.get_fdata(dtype=np.float32) > 0))
        masked_nonzero = int(np.count_nonzero(masked))
        total = int(np.prod(img.shape[:3]))
        if mask_nonzero == 0 or masked_nonzero == 0:
            raise ValueError("Mask or masked image is empty")

        out["mask_nonzero_voxels"] = str(mask_nonzero)
        out["mask_fraction"] = f"{mask_nonzero / total:.8g}"
        out["masked_nonzero_voxels"] = str(masked_nonzero)
        out["masked_input_sha256"] = sha256_file(output_path)
        out["status"] = "ok"
    except Exception as exc:
        out["error"] = repr(exc)
    return out


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--status-csv", required=True, type=Path)
    parser.add_argument("--infer-fastsurfer-label", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.input_manifest)
    if args.limit > 0:
        rows = rows[: args.limit]

    status_rows = [
        process_row(row, index, args.output_dir, args.infer_fastsurfer_label, args.overwrite)
        for index, row in enumerate(rows, start=1)
    ]
    ok_rows = [row for row in status_rows if row["status"] == "ok"]
    write_csv(args.status_csv, status_rows)
    write_csv(args.output_manifest, ok_rows, fieldnames=list(status_rows[0].keys()) if status_rows else ["input"])
    print(f"Prepared {len(ok_rows)}/{len(status_rows)} NeuroFM masked inputs")


if __name__ == "__main__":
    main()
