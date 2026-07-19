#!/usr/bin/env python3
"""Validate NeuroFM named brain-health columns against a direct API output."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_KEYS = ["brain_age", "sex", "ventricle_volume", "brain_volume"]


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurofm-repo", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.neurofm_repo))
    from neurofm import NeuroFM  # noqa: PLC0415
    from neurofm.io import save_batch_summary  # noqa: PLC0415
    from neurofm.model import BRAIN_HEALTH_KEYS  # noqa: PLC0415

    if list(BRAIN_HEALTH_KEYS) != EXPECTED_KEYS:
        raise ValueError(f"Unexpected NeuroFM code-level schema: {BRAIN_HEALTH_KEYS}")

    model = NeuroFM(
        variant="neurofm-s",
        device="cpu",
        weights=str(args.weights),
        cache_dir=str(args.weights.parent),
    )
    result = model.predict(str(args.input), outputs=["brain_health"])
    direct = np.asarray(result["brain_health"], dtype=float).reshape(-1)
    if direct.shape != (4,) or not np.isfinite(direct).all():
        raise ValueError(f"Invalid direct NeuroFM brain_health output: {direct}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    save_batch_summary(
        [{"brain_health": direct}],
        [str(args.input)],
        args.work_dir,
        ["brain_health"],
    )
    summary = pd.read_csv(args.work_dir / "results_summary.csv")
    actual_columns = [column for column in summary.columns if column != "input"]
    if actual_columns != EXPECTED_KEYS:
        raise ValueError(f"Unexpected summary columns: {actual_columns}")
    named = summary.loc[0, EXPECTED_KEYS].to_numpy(dtype=float)
    if not np.allclose(direct, named, rtol=1e-6, atol=1e-6):
        raise ValueError(f"Direct and named outputs differ: {direct} vs {named}")

    payload = {
        "schema_status": "pass",
        "code_level_keys": EXPECTED_KEYS,
        "direct_output": direct.tolist(),
        "named_summary_output": named.tolist(),
        "max_absolute_difference": float(np.max(np.abs(direct - named))),
        "weights_sha256": sha256_file(args.weights),
        "input_sha256": sha256_file(args.input),
        "interpretation": (
            "This validates output naming and execution only. It does not validate "
            "age, sex, segmentation, morphometry, or health claims."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
