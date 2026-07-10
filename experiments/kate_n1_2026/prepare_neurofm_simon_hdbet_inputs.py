#!/usr/bin/env python3
"""Prepare SIMON all-orig HD-BET skull-stripped inputs for NeuroFM.

This branch is intentionally separate from the raw-orig diagnostic branch.
NeuroFM's age/brain-health inference expects skull-stripped T1 inputs; the
FastSurfer ``orig.mgz`` files used here are available local source derivatives,
not untouched scanner-native SIMON data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_DATA_ROOT = Path(
    "/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_DATA_ROOT
    / "reprocessed_2026"
    / "foundation_models"
    / "neurofm_simon_hdbet"
)
DEFAULT_REPO_DATA_DIR = Path("data/kate_n1_2026")
DEFAULT_INPUT_MANIFEST = Path("experiments/kate_n1_2026/midi_brainage_simon_all_orig_inputs.csv")
DEFAULT_HD_BET = Path("/home/kate/.venvs/midi_brainage_py311/bin/hd-bet")


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


def shape_text(values: tuple[int, ...]) -> str:
    return "x".join(str(v) for v in values)


def zoom_text(img: nib.spatialimages.SpatialImage) -> str:
    return "x".join(f"{float(v):.6g}" for v in img.header.get_zooms()[:3])


def safe_stem(scan_id: str, index: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in scan_id)
    return safe or f"scan_{index:04d}"


def row_image_path(row: dict[str, str]) -> Path:
    for key in ("image_path", "path", "input", "input_path_wsl"):
        value = row.get(key, "")
        if value:
            return Path(value)
    raise KeyError("Input row must contain one of image_path/path/input/input_path_wsl")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def image_summary(path: Path) -> dict[str, str]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    return {
        "shape": shape_text(img.shape[:3]),
        "zooms_mm": zoom_text(img),
        "dtype": str(data.dtype),
        "nonzero_voxels": str(int(np.count_nonzero(data))),
        "size_bytes": str(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def mask_summary(path: Path) -> dict[str, str]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    nonzero = int(np.count_nonzero(data))
    total = int(np.prod(img.shape[:3]))
    return {
        "shape": shape_text(img.shape[:3]),
        "zooms_mm": zoom_text(img),
        "nonzero_voxels": str(nonzero),
        "fraction": f"{nonzero / total:.8g}" if total else "",
        "size_bytes": str(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def convert_mgz_to_nifti(source_path: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return
    img = nib.load(str(source_path))
    data = np.asanyarray(img.dataobj)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, img.affine), str(output_path))


def run_hdbet(
    hd_bet: Path,
    converted_path: Path,
    output_path: Path,
    log_path: Path,
    device: str,
    disable_tta: bool,
    overwrite: bool,
) -> tuple[str, str, float]:
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return "skipped_existing", "", 0.0

    command = [
        str(hd_bet),
        "-i",
        str(converted_path),
        "-o",
        str(output_path),
        "-device",
        device,
        "--save_bet_mask",
    ]
    if disable_tta:
        command.append("--disable_tta")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("command: " + " ".join(command) + "\n")
        proc = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - start
    if proc.returncode != 0:
        return "failed", f"hd-bet exited with rc={proc.returncode}; log={log_path}", elapsed
    return "ok", "", elapsed


def process_row(
    row: dict[str, str],
    index: int,
    converted_dir: Path,
    hdbet_dir: Path,
    logs_dir: Path,
    hd_bet: Path,
    device: str,
    disable_tta: bool,
    overwrite: bool,
    hd_bet_version: str,
) -> dict[str, str]:
    source_path = row_image_path(row)
    scan_id = row.get("scan_id", "") or source_path.name.removesuffix(".mgz")
    stem = safe_stem(scan_id, index)
    converted_path = converted_dir / f"{stem}_orig_for_hdbet.nii.gz"
    output_path = hdbet_dir / f"{stem}_hdbet.nii.gz"
    mask_path = hdbet_dir / f"{stem}_hdbet_bet.nii.gz"
    log_path = logs_dir / f"{stem}_hdbet.log"

    out = dict(row)
    out.update(
        {
            "method": "HD_BET",
            "source_image": str(source_path),
            "converted_input": str(converted_path),
            "input": str(output_path),
            "mask_path": str(mask_path),
            "status": "failed",
            "error": "",
            "hd_bet_version": hd_bet_version,
            "device": device,
            "disable_tta": "1" if disable_tta else "0",
            "elapsed_seconds": "",
            "input_shape": "",
            "converted_shape": "",
            "output_shape": "",
            "mask_shape": "",
            "input_zooms_mm": "",
            "converted_zooms_mm": "",
            "output_zooms_mm": "",
            "mask_zooms_mm": "",
            "input_dtype": "",
            "converted_dtype": "",
            "output_dtype": "",
            "input_nonzero_voxels": "",
            "converted_nonzero_voxels": "",
            "output_nonzero_voxels": "",
            "mask_nonzero_voxels": "",
            "mask_fraction": "",
            "input_sha256": "",
            "converted_sha256": "",
            "output_sha256": "",
            "mask_sha256": "",
            "input_size_bytes": "",
            "converted_size_bytes": "",
            "output_size_bytes": "",
            "mask_size_bytes": "",
            "log_path": str(log_path),
            "preprocessing_level": "fastsurfer_orig_mgz_to_nifti_hdbet_skullstrip",
            "preprocessing_detail": "mgz_to_nifti_then_hdbet_skullstrip_cpu_disable_tta",
            "claim_level": "input_preprocessing_only_not_age_claim",
            "notes": (
                "SIMON FastSurfer orig.mgz local derivative converted to NIfTI, then skull-stripped "
                "with HD-BET for NeuroFM. This is not raw scanner-native SIMON evidence."
            ),
        }
    )

    try:
        if not hd_bet.exists():
            raise FileNotFoundError(f"Missing HD-BET executable: {hd_bet}")
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source image: {source_path}")

        src = image_summary(source_path)
        out["input_shape"] = src["shape"]
        out["input_zooms_mm"] = src["zooms_mm"]
        out["input_dtype"] = src["dtype"]
        out["input_nonzero_voxels"] = src["nonzero_voxels"]
        out["input_sha256"] = src["sha256"]
        out["input_size_bytes"] = src["size_bytes"]

        convert_mgz_to_nifti(source_path, converted_path, overwrite)
        conv = image_summary(converted_path)
        out["converted_shape"] = conv["shape"]
        out["converted_zooms_mm"] = conv["zooms_mm"]
        out["converted_dtype"] = conv["dtype"]
        out["converted_nonzero_voxels"] = conv["nonzero_voxels"]
        out["converted_sha256"] = conv["sha256"]
        out["converted_size_bytes"] = conv["size_bytes"]

        hdbet_status, hdbet_error, elapsed = run_hdbet(
            hd_bet=hd_bet,
            converted_path=converted_path,
            output_path=output_path,
            log_path=log_path,
            device=device,
            disable_tta=disable_tta,
            overwrite=overwrite,
        )
        out["elapsed_seconds"] = f"{elapsed:.3f}"
        if hdbet_status == "failed":
            raise RuntimeError(hdbet_error)

        if not output_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Missing HD-BET output or mask: {output_path}, {mask_path}")

        result = image_summary(output_path)
        mask = mask_summary(mask_path)
        if int(result["nonzero_voxels"]) == 0 or int(mask["nonzero_voxels"]) == 0:
            raise ValueError("HD-BET output or mask is empty")

        out["output_shape"] = result["shape"]
        out["output_zooms_mm"] = result["zooms_mm"]
        out["output_dtype"] = result["dtype"]
        out["output_nonzero_voxels"] = result["nonzero_voxels"]
        out["output_sha256"] = result["sha256"]
        out["output_size_bytes"] = result["size_bytes"]
        out["mask_shape"] = mask["shape"]
        out["mask_zooms_mm"] = mask["zooms_mm"]
        out["mask_nonzero_voxels"] = mask["nonzero_voxels"]
        out["mask_fraction"] = mask["fraction"]
        out["mask_sha256"] = mask["sha256"]
        out["mask_size_bytes"] = mask["size_bytes"]
        out["status"] = "ok" if hdbet_status != "skipped_existing" else "ok_existing"
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", default=DEFAULT_INPUT_MANIFEST, type=Path)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=Path)
    parser.add_argument("--converted-dir", type=Path, default=None)
    parser.add_argument("--hdbet-dir", type=Path, default=None)
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument(
        "--output-manifest",
        default=DEFAULT_REPO_DATA_DIR / "neurofm_simon_hdbet_inputs_resolved.csv",
        type=Path,
    )
    parser.add_argument(
        "--status-csv",
        default=DEFAULT_REPO_DATA_DIR / "neurofm_simon_hdbet_preprocessing_status.csv",
        type=Path,
    )
    parser.add_argument("--hd-bet", default=DEFAULT_HD_BET, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--disable-tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.input_manifest)
    if args.limit > 0:
        rows = rows[: args.limit]

    converted_dir = args.converted_dir or args.output_root / "raw_orig_nifti_inputs"
    hdbet_dir = args.hdbet_dir or args.output_root / "hdbet_inputs"
    logs_dir = args.logs_dir or args.output_root / "logs"
    hd_bet_version = package_version("HD-BET") or package_version("HD_BET") or "2.0.1"

    status_rows = [
        process_row(
            row=row,
            index=index,
            converted_dir=converted_dir,
            hdbet_dir=hdbet_dir,
            logs_dir=logs_dir,
            hd_bet=args.hd_bet,
            device=args.device,
            disable_tta=args.disable_tta,
            overwrite=args.overwrite,
            hd_bet_version=hd_bet_version,
        )
        for index, row in enumerate(rows, start=1)
    ]
    ok_rows = [row for row in status_rows if row["status"] in {"ok", "ok_existing"}]

    write_csv(args.status_csv, status_rows)
    write_csv(args.output_manifest, ok_rows, fieldnames=list(status_rows[0].keys()) if status_rows else ["input"])

    n_failed = len(status_rows) - len(ok_rows)
    print(f"Prepared {len(ok_rows)}/{len(status_rows)} SIMON HD-BET NeuroFM inputs")
    print(f"Status CSV: {args.status_csv}")
    print(f"Resolved manifest: {args.output_manifest}")
    if n_failed:
        print(f"Failed rows: {n_failed}")


if __name__ == "__main__":
    main()
