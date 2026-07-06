#!/usr/bin/env python3
"""Download official NeuroFM weights to an external cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


VARIANT_FILES = {
    "neurofm-s": "neurofm-s.h5",
    "neurofm-m": "neurofm-m.h5",
    "neurofm-l": "neurofm-l.h5",
}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="neurofm-s", choices=sorted(VARIANT_FILES))
    parser.add_argument("--repo-id", default="NeuroAI-UofG/NeuroFM")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    args = parser.parse_args()

    filename = VARIANT_FILES[args.variant]
    downloaded = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            cache_dir=str(args.cache_dir),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / filename
    if downloaded.resolve() != destination.resolve():
        shutil.copy2(downloaded, destination)

    metadata = {
        "repo_id": args.repo_id,
        "variant": args.variant,
        "filename": filename,
        "downloaded_path": str(downloaded),
        "local_weight_path": str(destination),
        "sha256": sha256_file(destination),
        "license_note": "NeuroFM weights are CC BY-NC-SA 4.0; keep outside git.",
    }
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
