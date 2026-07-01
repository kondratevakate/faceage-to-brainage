#!/usr/bin/env python3
"""Evaluate a test-time-augmentation label ensemble in a common space.

The manifest must point to label maps that were already inverse-resampled into
one target grid. This script deliberately does not run the segmenter; it scores
the label ensemble that a method-specific wrapper produced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_DATA_ROOT = Path("/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reprocessed_2026" / "tta_label_ensembles"
DEFAULT_MANIFEST = Path(__file__).with_name("tta_label_ensemble_inputs.schema.csv")
DEFAULT_SUMMARY = Path("data/kate_n1_2026/tta_label_ensemble_summary.csv")
DEFAULT_GLOBAL_SUMMARY = Path("data/kate_n1_2026/tta_label_ensemble_global_summary.csv")

REQUIRED_COLUMNS = {"augmentation_id", "method", "scan_id", "label_path", "include_in_vote"}


@dataclass(frozen=True)
class TTAInput:
    augmentation_id: str
    method: str
    scan_id: str
    label_path: Path
    include_in_vote: bool
    transform_id: str
    transform_family: str
    angle_deg: str
    notes: str


@dataclass(frozen=True)
class EvaluationResult:
    label_rows: list[dict[str, object]]
    global_rows: list[dict[str, object]]
    metadata: dict[str, object]


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def resolve_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return data_root / path


def read_manifest(path: Path, data_root: Path) -> list[TTAInput]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest is missing required columns {sorted(missing)}: {path}")
        rows = []
        for row in reader:
            if not any((value or "").strip() for value in row.values()):
                continue
            rows.append(
                TTAInput(
                    augmentation_id=row["augmentation_id"],
                    method=row["method"],
                    scan_id=row["scan_id"],
                    label_path=resolve_path(data_root, row["label_path"]),
                    include_in_vote=parse_bool(row["include_in_vote"]),
                    transform_id=row.get("transform_id", ""),
                    transform_family=row.get("transform_family", ""),
                    angle_deg=row.get("angle_deg", ""),
                    notes=row.get("notes", ""),
                )
            )
    return rows


def load_label(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    image = nib.load(str(path))
    data = np.rint(np.asarray(image.get_fdata(dtype=np.float32))).astype(np.int32)
    return image, data


def validate_common_space(images: list[nib.Nifti1Image], arrays: list[np.ndarray]) -> None:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"All label maps must have the same shape after inverse resampling, got {sorted(shapes)}")
    ref_affine = images[0].affine
    for image in images[1:]:
        if not np.allclose(ref_affine, image.affine, atol=1e-4):
            raise ValueError("All label maps must share the same affine/common space.")


def hard_vote(stack: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tie-to-background hard vote, max vote fraction, and entropy bits."""
    n_aug = stack.shape[0]
    flat = stack.reshape(n_aug, -1)
    best_count = np.zeros(flat.shape[1], dtype=np.uint16)
    best_label = np.zeros(flat.shape[1], dtype=np.int32)
    tied = np.zeros(flat.shape[1], dtype=bool)
    entropy = np.zeros(flat.shape[1], dtype=np.float32)

    for label in labels:
        counts = np.count_nonzero(flat == int(label), axis=0).astype(np.uint16)
        better = counts > best_count
        same_best = (counts == best_count) & (counts > 0) & (best_label != int(label))
        best_label[better] = int(label)
        best_count[better] = counts[better]
        tied = (tied & ~better) | same_best

        nonzero = counts > 0
        if np.any(nonzero):
            p = counts[nonzero].astype(np.float32) / float(n_aug)
            entropy[nonzero] -= p * np.log2(p)

    best_label[tied] = 0
    vote_fraction = best_count.astype(np.float32) / float(n_aug)
    shape = stack.shape[1:]
    return best_label.reshape(shape), vote_fraction.reshape(shape), entropy.reshape(shape)


def per_label_rows(
    rows: list[TTAInput],
    stack: np.ndarray,
    labels: np.ndarray,
    hard: np.ndarray,
    vote_fraction: np.ndarray,
    entropy: np.ndarray,
    voxel_volume_mm3: float,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    n_aug = stack.shape[0]
    method_counts = Counter(row.method for row in rows)
    scan_counts = Counter(row.scan_id for row in rows)
    method = next(iter(method_counts)) if len(method_counts) == 1 else "mixed"
    scan_id = next(iter(scan_counts)) if len(scan_counts) == 1 else "mixed"

    for label in labels:
        label = int(label)
        if label == 0:
            continue
        counts = np.count_nonzero(stack == label, axis=(1, 2, 3)).astype(np.float64)
        mean_voxels = float(np.mean(counts))
        std_voxels = float(np.std(counts, ddof=1)) if n_aug > 1 else 0.0
        cv_pct = 100.0 * std_voxels / mean_voxels if mean_voxels > 0 else math.nan
        consensus_mask = hard == label
        present_mask = np.any(stack == label, axis=0)
        consensus_voxels = int(np.count_nonzero(consensus_mask))
        present_voxels = int(np.count_nonzero(present_mask))
        if consensus_voxels:
            mean_vote_fraction = float(np.mean(vote_fraction[consensus_mask]))
            mean_entropy_bits = float(np.mean(entropy[consensus_mask]))
        else:
            mean_vote_fraction = math.nan
            mean_entropy_bits = math.nan
        out.append(
            {
                "method": method,
                "scan_id": scan_id,
                "label": label,
                "n_augments": n_aug,
                "mean_voxels": round(mean_voxels, 6),
                "std_voxels": round(std_voxels, 6),
                "cv_pct": round(cv_pct, 6) if not math.isnan(cv_pct) else "",
                "min_voxels": int(np.min(counts)),
                "max_voxels": int(np.max(counts)),
                "mean_volume_ml": round(mean_voxels * voxel_volume_mm3 / 1000.0, 6),
                "std_volume_ml": round(std_voxels * voxel_volume_mm3 / 1000.0, 6),
                "consensus_voxels": consensus_voxels,
                "present_any_augment_voxels": present_voxels,
                "mean_consensus_vote_fraction": round(mean_vote_fraction, 6)
                if not math.isnan(mean_vote_fraction)
                else "",
                "mean_consensus_entropy_bits": round(mean_entropy_bits, 6)
                if not math.isnan(mean_entropy_bits)
                else "",
            }
        )
    return out


def global_rows(
    rows: list[TTAInput],
    stack: np.ndarray,
    labels: np.ndarray,
    hard: np.ndarray,
    vote_fraction: np.ndarray,
    entropy: np.ndarray,
) -> list[dict[str, object]]:
    foreground = np.any(stack != 0, axis=0)
    n_foreground = int(np.count_nonzero(foreground))
    method_counts = Counter(row.method for row in rows)
    scan_counts = Counter(row.scan_id for row in rows)
    method = next(iter(method_counts)) if len(method_counts) == 1 else "mixed"
    scan_id = next(iter(scan_counts)) if len(scan_counts) == 1 else "mixed"
    if n_foreground:
        mean_vote_fraction = float(np.mean(vote_fraction[foreground]))
        mean_entropy_bits = float(np.mean(entropy[foreground]))
    else:
        mean_vote_fraction = math.nan
        mean_entropy_bits = math.nan
    return [
        {
            "method": method,
            "scan_id": scan_id,
            "n_augments": stack.shape[0],
            "shape": "x".join(str(v) for v in stack.shape[1:]),
            "n_labels_including_background": len(labels),
            "foreground_union_voxels": n_foreground,
            "hard_vote_foreground_voxels": int(np.count_nonzero(hard)),
            "mean_foreground_vote_fraction": round(mean_vote_fraction, 6)
            if not math.isnan(mean_vote_fraction)
            else "",
            "mean_foreground_entropy_bits": round(mean_entropy_bits, 6) if not math.isnan(mean_entropy_bits) else "",
            "augmentation_ids": ";".join(row.augmentation_id for row in rows),
        }
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_nifti(path: Path, data: np.ndarray, reference: nib.Nifti1Image, dtype: np.dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data.astype(dtype), reference.affine, reference.header)
    nib.save(image, str(path))


def run_evaluation(
    manifest: Path,
    data_root: Path,
    output_dir: Path,
    write_hard_vote: bool,
    write_vote_fraction: bool,
    write_entropy: bool,
) -> EvaluationResult:
    manifest_rows = read_manifest(manifest, data_root)
    rows = [row for row in manifest_rows if row.include_in_vote]
    if len(rows) < 2:
        raise ValueError("At least two include_in_vote=1 label maps are required for a TTA ensemble.")

    images: list[nib.Nifti1Image] = []
    arrays: list[np.ndarray] = []
    for row in rows:
        image, array = load_label(row.label_path)
        images.append(image)
        arrays.append(array)
    validate_common_space(images, arrays)

    stack = np.stack(arrays, axis=0)
    labels = np.unique(stack)
    hard, vote_fraction, entropy = hard_vote(stack, labels)
    voxel_volume_mm3 = float(abs(np.linalg.det(images[0].affine[:3, :3])))

    label_rows = per_label_rows(rows, stack, labels, hard, vote_fraction, entropy, voxel_volume_mm3)
    global_summary = global_rows(rows, stack, labels, hard, vote_fraction, entropy)

    if write_hard_vote:
        write_nifti(output_dir / "hard_vote.nii.gz", hard, images[0], np.int16)
    if write_vote_fraction:
        write_nifti(output_dir / "vote_fraction.nii.gz", vote_fraction, images[0], np.float32)
    if write_entropy:
        write_nifti(output_dir / "entropy_bits.nii.gz", entropy, images[0], np.float32)

    metadata = {
        "manifest": str(manifest),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "n_manifest_rows": len(manifest_rows),
        "n_included_rows": len(rows),
        "labels": [int(label) for label in labels.tolist()],
        "label_paths": [str(row.label_path) for row in rows],
        "interpretation_limit": "TTA agreement estimates perturbation stability, not anatomical accuracy.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tta_label_ensemble_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return EvaluationResult(label_rows=label_rows, global_rows=global_summary, metadata=metadata)


def build_self_test_inputs(root: Path) -> Path:
    affine = np.eye(4)
    arrays: list[np.ndarray] = []
    base = np.zeros((8, 8, 8), dtype=np.int16)
    base[2:5, 2:5, 2:5] = 1
    base[5:7, 1:4, 1:4] = 2
    arrays.append(base)

    variant_shift = np.zeros_like(base)
    variant_shift[2:5, 3:6, 2:5] = 1
    variant_shift[5:7, 1:4, 1:4] = 2
    arrays.append(variant_shift)

    variant_volume = np.zeros_like(base)
    variant_volume[2:6, 2:5, 2:5] = 1
    variant_volume[5:7, 1:4, 1:4] = 2
    arrays.append(variant_volume)

    label_paths = []
    for index, array in enumerate(arrays):
        path = root / f"selftest_aug{index}.nii.gz"
        nib.save(nib.Nifti1Image(array, affine), str(path))
        label_paths.append(path)

    manifest = root / "selftest_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "augmentation_id",
            "method",
            "scan_id",
            "label_path",
            "include_in_vote",
            "transform_id",
            "transform_family",
            "angle_deg",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, path in enumerate(label_paths):
            writer.writerow(
                {
                    "augmentation_id": f"selftest_aug{index}",
                    "method": "selftest",
                    "scan_id": "synthetic",
                    "label_path": str(path),
                    "include_in_vote": "1",
                    "transform_id": f"synthetic_{index}",
                    "transform_family": "synthetic",
                    "angle_deg": "",
                    "notes": "generated by --self-test",
                }
            )
    return manifest


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="tta_label_ensemble_selftest_") as tmp:
        root = Path(tmp)
        manifest = build_self_test_inputs(root)
        result = run_evaluation(
            manifest=manifest,
            data_root=root,
            output_dir=root / "out",
            write_hard_vote=True,
            write_vote_fraction=True,
            write_entropy=True,
        )
        label1 = next(row for row in result.label_rows if row["label"] == 1)
        global_row = result.global_rows[0]
        if float(label1["cv_pct"]) <= 0:
            raise AssertionError("Self-test expected non-zero label-1 volume CV.")
        if float(global_row["mean_foreground_entropy_bits"]) <= 0:
            raise AssertionError("Self-test expected non-zero foreground entropy.")
        print("Self-test passed.")
        print(f"Label rows: {len(result.label_rows)}")
        print(f"Global mean entropy bits: {global_row['mean_foreground_entropy_bits']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate inverse-resampled TTA label maps.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--global-summary-csv", type=Path, default=DEFAULT_GLOBAL_SUMMARY)
    parser.add_argument("--write-hard-vote", action="store_true")
    parser.add_argument("--write-vote-fraction", action="store_true")
    parser.add_argument("--write-entropy", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    result = run_evaluation(
        manifest=args.manifest,
        data_root=args.data_root,
        output_dir=args.output_dir,
        write_hard_vote=args.write_hard_vote,
        write_vote_fraction=args.write_vote_fraction,
        write_entropy=args.write_entropy,
    )
    write_csv(args.summary_csv, result.label_rows)
    write_csv(args.global_summary_csv, result.global_rows)
    print(f"Wrote label summary to {args.summary_csv}")
    print(f"Wrote global summary to {args.global_summary_csv}")
    print(f"Wrote metadata to {args.output_dir / 'tta_label_ensemble_metadata.json'}")


if __name__ == "__main__":
    main()
