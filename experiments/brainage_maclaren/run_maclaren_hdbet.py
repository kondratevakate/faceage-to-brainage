#!/usr/bin/env python3
"""Run pinned HD-BET preprocessing for the locked Maclaren T1w manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_DATASET_ROOT = Path(
    "/mnt/d/data/faceage-to-brainage/sourcedata/maclaren_ds000239/"
    "R1.0.1/ds000239_R1.0.1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/d/data/faceage-to-brainage/derivatives/hdbet/2.0.1/maclaren_ds000239/R1.0.1"
)
DEFAULT_MANIFEST = Path("data/brainage/maclaren_t1w_inclusion.csv")
DEFAULT_HD_BET = Path("/home/kate/.venvs/midi_brainage_py311/bin/hd-bet")
DEFAULT_HD_BET_PYTHON = Path("/home/kate/.venvs/midi_brainage_py311/bin/python")
DEFAULT_CHECKPOINT = Path("/home/kate/hd-bet_params/release_2.0.0/fold_all/checkpoint_final.pth")
EXPECTED_CHECKPOINT_SHA256 = "d31dc59b4c5fe0028070901870c44c0b526f48c507ce50804941344356df7b52"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def output_paths(output_dir: Path, source_path: Path) -> tuple[Path, Path]:
    output_path = output_dir / source_path.name
    mask_path = output_dir / f"{source_path.name[:-7]}_bet.nii.gz"
    return output_path, mask_path


def summarize_nifti(path: Path, is_mask: bool = False) -> dict[str, str]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    total = int(np.prod(image.shape[:3]))
    nonzero = int(np.count_nonzero(data))
    summary = {
        "shape": "x".join(str(v) for v in image.shape),
        "voxel_size_mm": "x".join(f"{float(v):.6g}" for v in image.header.get_zooms()[:3]),
        "orientation": "".join(nib.aff2axcodes(image.affine)),
        "dtype": str(image.get_data_dtype()),
        "nonzero_voxels": str(nonzero),
        "nonzero_fraction": f"{nonzero / total:.8g}" if total else "",
        "size_bytes": str(path.stat().st_size),
        "sha256": sha256_file(path),
    }
    if is_mask:
        values = np.unique(data)
        summary["binary_mask"] = "1" if set(values.tolist()).issubset({0, 1}) else "0"
    return summary


def make_link(link_path: Path, source_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() != source_path.resolve():
            raise ValueError(f"Existing symlink has wrong target: {link_path}")
        return
    if link_path.exists():
        raise FileExistsError(f"Refusing to replace non-symlink staging input: {link_path}")
    link_path.symlink_to(source_path)


def inspect_row(row: dict[str, str], dataset_root: Path, output_dir: Path) -> dict[str, str]:
    source_path = dataset_root / row["relative_path"]
    output_path, mask_path = output_paths(output_dir, source_path)
    result = dict(row)
    result.update(
        {
            "source_image_runtime": str(source_path),
            "input": str(output_path),
            "mask_path": str(mask_path),
            "preprocessing_method": "HD-BET",
            "preprocessing_version": "2.0.1",
            "preprocessing_device": "cpu",
            "tta_disabled": "1",
            "status": "pending",
            "error": "",
            "output_shape": "",
            "output_voxel_size_mm": "",
            "output_orientation": "",
            "output_dtype": "",
            "output_nonzero_voxels": "",
            "output_sha256": "",
            "output_size_bytes": "",
            "mask_nonzero_voxels": "",
            "mask_fraction": "",
            "mask_binary": "",
            "mask_sha256": "",
            "mask_size_bytes": "",
            "preprocessing_claim": "skullstrip_input_preparation_not_segmentation_validation",
        }
    )
    try:
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if sha256_file(source_path) != row["source_sha256"]:
            raise ValueError(f"Source SHA-256 changed: {source_path}")
        if not output_path.is_file() or not mask_path.is_file():
            return result
        output = summarize_nifti(output_path)
        mask = summarize_nifti(mask_path, is_mask=True)
        if output["shape"] != mask["shape"]:
            raise ValueError(f"Output/mask shape mismatch for {source_path.name}")
        if output["nonzero_voxels"] == "0" or mask["nonzero_voxels"] == "0":
            raise ValueError(f"Empty output or mask for {source_path.name}")
        if mask["binary_mask"] != "1":
            raise ValueError(f"Non-binary HD-BET mask for {source_path.name}")
        result.update(
            {
                "status": "ok",
                "output_shape": output["shape"],
                "output_voxel_size_mm": output["voxel_size_mm"],
                "output_orientation": output["orientation"],
                "output_dtype": output["dtype"],
                "output_nonzero_voxels": output["nonzero_voxels"],
                "output_sha256": output["sha256"],
                "output_size_bytes": output["size_bytes"],
                "mask_nonzero_voxels": mask["nonzero_voxels"],
                "mask_fraction": mask["nonzero_fraction"],
                "mask_binary": mask["binary_mask"],
                "mask_sha256": mask["sha256"],
                "mask_size_bytes": mask["size_bytes"],
            }
        )
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
    return result


def refresh_and_write(
    rows: list[dict[str, str]],
    dataset_root: Path,
    output_dir: Path,
    status_csv: Path,
    inference_manifest: Path,
) -> list[dict[str, str]]:
    status_rows = [inspect_row(row, dataset_root, output_dir) for row in rows]
    write_csv(status_csv, status_rows)
    ok_rows = [row for row in status_rows if row["status"] == "ok"]
    write_csv(inference_manifest, ok_rows)
    return status_rows


def chunks(rows: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hd-bet", type=Path, default=DEFAULT_HD_BET)
    parser.add_argument("--hd-bet-python", type=Path, default=DEFAULT_HD_BET_PYTHON)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    if not args.hd_bet.is_file() or not os.access(args.hd_bet, os.X_OK):
        raise FileNotFoundError(f"Missing HD-BET executable: {args.hd_bet}")
    if not args.hd_bet_python.is_file() or not os.access(args.hd_bet_python, os.X_OK):
        raise FileNotFoundError(f"Missing HD-BET Python: {args.hd_bet_python}")
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected HD-BET checkpoint SHA-256: {checkpoint_hash}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    rows = [row for row in read_rows(args.manifest) if row["include"] == "1"]
    if len(rows) != 120:
        raise ValueError(f"Locked manifest must contain 120 included rows, found {len(rows)}")
    if args.limit > 0:
        rows = rows[: args.limit]

    output_dir = args.output_root / "hdbet_inputs"
    invocation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_input_root = args.output_root / "batch_inputs" / invocation_id
    logs_dir = args.output_root / "logs"
    status_csv = args.output_root / "preprocessing_status.csv"
    inference_manifest = args.output_root / "neurofm_inputs.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    status_rows = refresh_and_write(
        rows, args.dataset_root, output_dir, status_csv, inference_manifest
    )
    pending = [row for row, status in zip(rows, status_rows) if status["status"] != "ok"]
    pending_batches = chunks(pending, args.batch_size)
    if args.max_batches > 0:
        pending_batches = pending_batches[: args.max_batches]

    run_start = time.time()
    for batch_index, batch in enumerate(pending_batches, start=1):
        batch_dir = batch_input_root / f"batch_{batch_index:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        for row in batch:
            source_path = args.dataset_root / row["relative_path"]
            make_link(batch_dir / source_path.name, source_path)

        log_path = logs_dir / f"batch_{batch_index:04d}.log"
        low_memory_wrapper = Path(__file__).with_name("hdbet_low_memory.py")
        command = [
            str(args.hd_bet_python),
            str(low_memory_wrapper),
            "--input-folder",
            str(batch_dir),
            "--output-folder",
            str(output_dir),
            "--preprocessing-workers",
            "1",
            "--export-workers",
            "1",
        ]
        print(f"HD-BET batch {batch_index}/{len(pending_batches)}: {len(batch)} scan(s)")
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write("command: " + " ".join(command) + "\n")
            process = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.time() - started
        print(f"HD-BET batch {batch_index} rc={process.returncode} elapsed={elapsed:.1f}s")
        status_rows = refresh_and_write(
            rows, args.dataset_root, output_dir, status_csv, inference_manifest
        )
        if process.returncode != 0:
            raise RuntimeError(f"HD-BET batch failed; see {log_path}")

    status_rows = refresh_and_write(
        rows, args.dataset_root, output_dir, status_csv, inference_manifest
    )
    n_ok = sum(row["status"] == "ok" for row in status_rows)
    metadata = {
        "dataset_id": "maclaren_ds000239",
        "release": "R1.0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hd_bet_version": importlib.metadata.version("HD-BET"),
        "hd_bet_checkpoint_sha256": checkpoint_hash,
        "device": "cpu",
        "tta_disabled": True,
        "low_memory_preprocessing_workers": 1,
        "low_memory_export_workers": 1,
        "batch_size": args.batch_size,
        "n_selected": len(rows),
        "n_ok": n_ok,
        "n_failed_or_pending": len(rows) - n_ok,
        "elapsed_seconds_this_invocation": time.time() - run_start,
        "status_csv": str(status_csv),
        "inference_manifest": str(inference_manifest),
        "claim": "preprocessing_qc_only_not_segmentation_validation",
    }
    (args.output_root / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"HD-BET ready: {n_ok}/{len(rows)}; manifest: {inference_manifest}")


if __name__ == "__main__":
    main()
