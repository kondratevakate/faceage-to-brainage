#!/usr/bin/env python3
"""Summarize BrainChop smoke-test runtime results and compact label stats."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_OUTPUT_CSV = Path("data/kate_n1_2026/brainchop_0.2.5_smoke_results.csv")


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["summary_path"] = str(path)
                rows.append(row)
    return rows


def label_stats(path_value: str, status: str) -> dict[str, str]:
    path = Path(path_value)
    if status != "done" or not path.exists():
        return {
            "output_exists": "0",
            "shape": "",
            "voxel_size_mm": "",
            "n_nonzero_voxels": "",
            "n_labels": "",
            "labels": "",
        }
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    labels = sorted(int(value) for value in np.unique(data) if int(value) != 0)
    return {
        "output_exists": "1",
        "shape": "x".join(str(dim) for dim in image.shape),
        "voxel_size_mm": "x".join(f"{float(value):.4g}" for value in image.header.get_zooms()[:3]),
        "n_nonzero_voxels": str(int(np.count_nonzero(data))),
        "n_labels": str(len(labels)),
        "labels": ";".join(str(label) for label in labels),
    }


def write_output(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "scan_id",
        "session",
        "model",
        "status",
        "elapsed_sec",
        "returncode",
        "output_exists",
        "shape",
        "voxel_size_mm",
        "n_nonzero_voxels",
        "n_labels",
        "labels",
        "output_path",
        "log_path",
        "note",
        "summary_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            stats = label_stats(row["output_path"], row["status"])
            merged = {**row, **stats}
            writer.writerow({key: merged.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    rows = read_rows(args.summary_csv)
    write_output(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
