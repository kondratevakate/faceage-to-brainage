#!/usr/bin/env python3
"""Resolve a candidate-column input manifest without tracking raw MRI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def materialize(src: Path, dst: Path, mode: str) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return dst
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return dst
        except OSError:
            shutil.copy2(src, dst)
            return dst
    if mode == "none":
        return src
    raise ValueError(f"Unsupported materialize mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--candidate-column", required=True)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--materialize-dir", type=Path)
    parser.add_argument("--materialize", choices=["none", "symlink", "copy"], default="none")
    parser.add_argument("--hash", action="store_true")
    args = parser.parse_args()

    out_rows = []
    for row in read_rows(args.input_manifest):
        if row.get(args.candidate_column, "0") != "1":
            continue
        src = args.data_root / row["relative_path"]
        if not src.exists():
            raise FileNotFoundError(f"Missing input for {row['scan_id']}: {src}")
        used_path = src
        if args.materialize != "none":
            if args.materialize_dir is None:
                raise ValueError("--materialize-dir is required unless --materialize none")
            used_path = materialize(src, args.materialize_dir / f"{row['scan_id']}.nii.gz", args.materialize)
        out_rows.append(
            {
                "scan_id": row["scan_id"],
                "session": row["session"],
                "modality_hint": row["modality_hint"],
                "analysis_role": row["analysis_role"],
                "path": str(used_path),
                "source_relative_path": row["relative_path"],
                "source_sha256": sha256_file(src) if args.hash else "",
                "notes": row.get("notes", ""),
            }
        )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "scan_id",
            "session",
            "modality_hint",
            "analysis_role",
            "path",
            "source_relative_path",
            "source_sha256",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {args.output_manifest}")


if __name__ == "__main__":
    main()
