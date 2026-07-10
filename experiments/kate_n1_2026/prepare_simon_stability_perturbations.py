#!/usr/bin/env python3
"""Prepare SIMON perturbation inputs for brain-age model stability checks.

The perturbations are QC probes, not biologically valid augmentations:
- brain_size_* changes the apparent brain content scale in a fixed grid;
- resample_roundtrip_* adds small interpolation/resampling blur;
- rotate_z_* rotates the volume in-plane in a fixed grid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


DEFAULT_REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = DEFAULT_REPO / "data" / "kate_n1_2026"
DEFAULT_SOURCE = DEFAULT_DATA / "neurofm_simon_fastsurfer_mask_inputs_resolved.csv"
DEFAULT_SEED = Path(__file__).resolve().parent / "midi_brainage_simon_stratified12_inputs.csv"
DEFAULT_OUT_ROOT = Path(
    "/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years/"
    "reprocessed_2026/foundation_models/simon_model_stability"
)


PERTURBATIONS = [
    ("baseline", 0.0),
    ("brain_size_scale_0p98", 0.98),
    ("brain_size_scale_1p02", 1.02),
    ("resample_roundtrip_0p98", 0.98),
    ("resample_roundtrip_1p02", 1.02),
    ("rotate_z_minus2deg", -2.0),
    ("rotate_z_plus2deg", 2.0),
]


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


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_")


def select_subset(source_rows: list[dict[str, str]], seed_rows: list[dict[str, str]], n: int) -> list[dict[str, str]]:
    by_scan = {row["scan_id"]: row for row in source_rows}
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for seed in seed_rows:
        scan_id = seed.get("scan_id", "")
        if scan_id in by_scan and scan_id not in seen:
            selected.append(by_scan[scan_id])
            seen.add(scan_id)
    if len(selected) >= n:
        return selected[:n]

    candidates = sorted(
        (row for row in source_rows if row["scan_id"] not in seen),
        key=lambda r: (float(r.get("chronological_age_years") or 0), r["scan_id"]),
    )
    if not candidates:
        return selected

    needed = n - len(selected)
    if needed >= len(candidates):
        selected.extend(candidates)
        return selected

    picks = np.linspace(0, len(candidates) - 1, num=needed)
    for idx in picks:
        row = candidates[int(round(idx))]
        if row["scan_id"] not in seen:
            selected.append(row)
            seen.add(row["scan_id"])
    return selected


def center_crop_or_pad(data: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    src_slices = []
    dst_slices = []
    for src_len, dst_len in zip(data.shape, shape):
        if src_len >= dst_len:
            start = (src_len - dst_len) // 2
            src_slices.append(slice(start, start + dst_len))
            dst_slices.append(slice(0, dst_len))
        else:
            start = (dst_len - src_len) // 2
            src_slices.append(slice(0, src_len))
            dst_slices.append(slice(start, start + src_len))
    out[tuple(dst_slices)] = data[tuple(src_slices)]
    return out


def brain_size_scale(data: np.ndarray, scale: float) -> np.ndarray:
    shape = np.asarray(data.shape, dtype=np.float64)
    center = (shape - 1.0) / 2.0
    matrix = np.eye(3) / scale
    offset = center - matrix @ center
    return ndimage.affine_transform(
        data,
        matrix=matrix,
        offset=offset,
        output_shape=data.shape,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    ).astype(np.float32, copy=False)


def resample_roundtrip(data: np.ndarray, factor: float) -> np.ndarray:
    first = ndimage.zoom(data, zoom=factor, order=3, mode="constant", cval=0.0, prefilter=True)
    back_factor = np.asarray(data.shape, dtype=np.float64) / np.asarray(first.shape, dtype=np.float64)
    second = ndimage.zoom(first, zoom=back_factor, order=3, mode="constant", cval=0.0, prefilter=True)
    return center_crop_or_pad(second.astype(np.float32, copy=False), data.shape)


def rotate_z(data: np.ndarray, angle: float) -> np.ndarray:
    return ndimage.rotate(
        data,
        angle=angle,
        axes=(0, 1),
        reshape=False,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    ).astype(np.float32, copy=False)


def perturb(data: np.ndarray, kind: str, value: float) -> np.ndarray:
    if kind == "baseline":
        return data
    if kind.startswith("brain_size_scale"):
        return brain_size_scale(data, value)
    if kind.startswith("resample_roundtrip"):
        return resample_roundtrip(data, value)
    if kind.startswith("rotate_z"):
        return rotate_z(data, value)
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_DATA / "simon_stability_perturbation_inputs.csv")
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_DATA / "simon_stability_perturbation_status.csv")
    parser.add_argument("--subset-size", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_rows = read_rows(args.source_manifest)
    seed_rows = read_rows(args.seed_manifest)
    subset = select_subset(source_rows, seed_rows, args.subset_size)
    output_dir = args.output_root / "perturbed_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    status_rows: list[dict[str, str]] = []
    for source in subset:
        base_path = Path(source["input"])
        img = nib.load(str(base_path))
        data = img.get_fdata(dtype=np.float32)
        base_hash = sha256_file(base_path)
        for kind, value in PERTURBATIONS:
            out = dict(source)
            stem = f"{safe_name(source['scan_id'])}_{kind}"
            out_path = base_path if kind == "baseline" else output_dir / f"{stem}.nii.gz"
            out.update(
                {
                    "source_scan_id": source["scan_id"],
                    "perturbation": kind,
                    "perturbation_value": f"{value:.6g}",
                    "base_input": str(base_path),
                    "base_input_sha256": base_hash,
                    "input": str(out_path),
                    "path": str(out_path),
                    "stability_branch": "simon_neurofm_masked_subset_perturbations",
                    "status": "failed",
                    "error": "",
                    "input_sha256": "",
                    "preprocessing_detail": "stability_qc_perturbation_on_existing_masked_neurofm_input",
                }
            )
            try:
                if kind != "baseline" and (args.overwrite or not out_path.exists()):
                    new_data = perturb(data, kind, value)
                    nib.save(nib.Nifti1Image(new_data, img.affine, img.header), str(out_path))
                if not out_path.exists():
                    raise FileNotFoundError(out_path)
                out["input_sha256"] = sha256_file(out_path)
                out["status"] = "ok"
            except Exception as exc:
                out["error"] = repr(exc)
            status_rows.append(out)

    ok_rows = [row for row in status_rows if row["status"] == "ok"]
    write_csv(args.status_csv, status_rows)
    write_csv(args.output_manifest, ok_rows, fieldnames=list(status_rows[0].keys()) if status_rows else ["input"])
    print(f"Prepared {len(ok_rows)}/{len(status_rows)} perturbation inputs")
    print(f"Subset scans: {', '.join(row['scan_id'] for row in subset)}")


if __name__ == "__main__":
    main()
