#!/usr/bin/env python3
"""Run BrainFM inference on Kate n=1 inputs and save QC-friendly outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pooled_feature_stats(features: list[torch.Tensor]) -> dict[str, float]:
    stats: dict[str, float] = {}
    for level, feat in enumerate(features):
        arr = feat.detach().float().cpu()
        if arr.ndim < 3:
            flat = arr.reshape(arr.shape[0], -1)
        else:
            flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
        channel_mean = flat.mean(dim=-1).squeeze(0).numpy()
        channel_std = flat.std(dim=-1).squeeze(0).numpy()
        for idx, value in enumerate(np.atleast_1d(channel_mean)):
            stats[f"feat_l{level}_mean_{idx:04d}"] = float(value)
        for idx, value in enumerate(np.atleast_1d(channel_std)):
            stats[f"feat_l{level}_std_{idx:04d}"] = float(value)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brainfm-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--win-size", default="160,160,160")
    parser.add_argument("--stride", default="80,80,80")
    parser.add_argument("--write-volumes", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing BrainFM checkpoint: {args.checkpoint}")
    if not (args.brainfm_repo / "utils" / "test_utils.py").exists():
        raise FileNotFoundError(f"Missing BrainFM repo: {args.brainfm_repo}")

    sys.path.insert(0, str(args.brainfm_repo))
    prev_cwd = Path.cwd()
    os.chdir(args.brainfm_repo)
    try:
        import utils.test_utils as test_utils
        from utils.misc import viewVolume
    finally:
        os.chdir(prev_cwd)

    # The current BrainFM repo tree uses cfgs/... while test_utils has stale cfg/defaults paths.
    test_utils.default_gen_cfg_file = str(args.brainfm_repo / "cfgs" / "generator" / "default.yaml")
    test_utils.default_train_cfg_file = str(args.brainfm_repo / "cfgs" / "trainer" / "default_train.yaml")
    test_utils.default_val_file = str(args.brainfm_repo / "cfgs" / "trainer" / "default_val.yaml")
    test_utils.gen_cfg_dir = ""
    test_utils.train_cfg_dir = ""
    test_utils.atlas_path = str(args.brainfm_repo / "files" / "gca.mgz")

    gen_cfg = str(args.brainfm_repo / "cfgs" / "generator" / "test" / "demo_test.yaml")
    model_cfg = str(args.brainfm_repo / "cfgs" / "trainer" / "test" / "demo_test.yaml")
    win_size = [int(x) for x in args.win_size.split(",")] if args.win_size.lower() != "none" else None

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda:0" if use_cuda else "cpu")
    rows = read_manifest(args.input_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gen_args = test_utils.utils.preprocess_cfg([test_utils.default_gen_cfg_file, gen_cfg], cfg_dir=test_utils.gen_cfg_dir)
    train_args = test_utils.utils.preprocess_cfg(
        [test_utils.default_train_cfg_file, test_utils.default_val_file, model_cfg],
        cfg_dir=test_utils.train_cfg_dir,
    )
    gen_args, train_args, feat_model, processors, _criterion, postprocessor = test_utils.build_model(
        gen_args,
        train_args,
        device,
    )
    test_utils.load_checkpoint(str(args.checkpoint), [feat_model], model_keys=["model"], to_print=False)
    feat_model.eval()

    feature_rows = []
    volume_rows = []
    with torch.no_grad():
        for row in rows:
            scan_id = row["scan_id"]
            img_path = Path(row["path"])
            scan_dir = args.output_dir / scan_id
            scan_dir.mkdir(parents=True, exist_ok=True)

            im, _orig, _high_res, _bf, aff, _crop_start, orig_shape = test_utils.prepare_image(
                str(img_path),
                win_size=win_size,
                zero_crop_first=True,
                spacing=None,
                im_only=False,
                add_bf=False,
                device=device,
            )
            samples = [{"input": im}]
            outputs, _ = feat_model(samples)
            for processor in processors:
                outputs = processor(outputs, samples)
            if postprocessor is not None:
                outputs, _, _ = postprocessor(
                    gen_args,
                    train_args,
                    outputs,
                    samples,
                    target=None,
                    feats=None,
                    tasks=gen_args.tasks,
                )
            outs = outputs[0]

            feature_stats = pooled_feature_stats(outs.get("feat", []))
            feature_rows.append(
                {
                    "scan_id": scan_id,
                    "session": row.get("session", ""),
                    "modality_hint": row.get("modality_hint", ""),
                    "analysis_role": row.get("analysis_role", ""),
                    "image_path": str(img_path),
                    "image_sha256": sha256_file(img_path),
                    "prepared_shape": "x".join(str(x) for x in im.shape[2:]),
                    "original_shape_after_orientation": "x".join(str(x) for x in orig_shape),
                    **feature_stats,
                }
            )

            if args.write_volumes:
                mask = im.clone()
                mask[im != 0.0] = 1.0
                for key, value in outs.items():
                    if "feat" in key or "segmentation" in key:
                        continue
                    out_name = f"out_{key}"
                    viewVolume(value * mask, aff, names=[out_name], save_dir=str(scan_dir))
                    volume_rows.append({"scan_id": scan_id, "output_key": key, "path": str(scan_dir / f"{out_name}.nii.gz")})

    features_csv = args.output_dir / "brainfm_feature_summaries.csv"
    with features_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)

    volumes_csv = args.output_dir / "brainfm_volume_outputs.csv"
    with volumes_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["scan_id", "output_key", "path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(volume_rows)

    metadata = {
        "method": "BrainFM",
        "brainfm_repo": str(args.brainfm_repo),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "device": str(device),
        "n_images": len(feature_rows),
        "write_volumes": bool(args.write_volumes),
        "features_csv": str(features_csv),
        "volumes_csv": str(volumes_csv),
    }
    (args.output_dir / "brainfm_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote BrainFM outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
