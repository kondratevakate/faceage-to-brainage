#!/usr/bin/env python3
"""Header-affine spatial pseudo-GT pilot for 2024 label maps.

This is a lightweight spatial stage. It resamples label maps into one target
grid using NIfTI header affines and nearest-neighbor interpolation, then builds
hard-vote pseudo-GT labels from trusted sources.

It is not a substitute for deformable/subject-template registration. The output
is useful as a fast QC pilot and as a reproducible bridge to the later template
space pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


ASEG_LABELS = {
    2: "left_cerebral_white_matter",
    3: "left_cerebral_cortex",
    4: "left_lateral_ventricle",
    5: "left_inferior_lateral_ventricle",
    7: "left_cerebellum_white_matter",
    8: "left_cerebellum_cortex",
    10: "left_thalamus",
    11: "left_caudate",
    12: "left_putamen",
    13: "left_pallidum",
    14: "third_ventricle",
    15: "fourth_ventricle",
    16: "brain_stem",
    17: "left_hippocampus",
    18: "left_amygdala",
    24: "csf",
    26: "left_accumbens_area",
    28: "left_ventral_dc",
    41: "right_cerebral_white_matter",
    42: "right_cerebral_cortex",
    43: "right_lateral_ventricle",
    44: "right_inferior_lateral_ventricle",
    46: "right_cerebellum_white_matter",
    47: "right_cerebellum_cortex",
    49: "right_thalamus",
    50: "right_caudate",
    51: "right_putamen",
    52: "right_pallidum",
    53: "right_hippocampus",
    54: "right_amygdala",
    58: "right_accumbens_area",
    60: "right_ventral_dc",
}


@dataclass(frozen=True)
class Source:
    source_id: str
    method: str
    session_group: str
    scan_id: str
    path: Path
    include_in_reference: bool
    reference_grid: bool
    notes: str


def read_manifest(path: Path, data_root: Path) -> list[Source]:
    sources: list[Source] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sources.append(
                Source(
                    source_id=row["source_id"],
                    method=row["method"],
                    session_group=row["session_group"],
                    scan_id=row["scan_id"],
                    path=data_root / row["label_path"],
                    include_in_reference=row.get("include_in_reference", "0") == "1",
                    reference_grid=row.get("reference_grid", "0") == "1",
                    notes=row.get("notes", ""),
                )
            )
    return sources


def load_target(sources: list[Source]) -> tuple[Source, nib.Nifti1Image]:
    grid_sources = [source for source in sources if source.reference_grid]
    if len(grid_sources) != 1:
        raise ValueError(f"Expected exactly one reference_grid=1 source, got {len(grid_sources)}")
    target_source = grid_sources[0]
    return target_source, nib.load(str(target_source.path))


def resample_label_map(path: Path, target_img: nib.Nifti1Image) -> np.ndarray:
    img = nib.load(str(path))
    if img.shape == target_img.shape and np.allclose(img.affine, target_img.affine):
        data = np.asarray(img.get_fdata(dtype=np.float32))
    else:
        data = np.asarray(resample_from_to(img, target_img, order=0).get_fdata(dtype=np.float32))
    return np.rint(data).astype(np.int16)


def build_consensus(
    arrays: list[np.ndarray],
    min_votes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack(arrays, axis=0)
    consensus = np.zeros(stack.shape[1:], dtype=np.int16)
    max_vote_fraction = np.zeros(stack.shape[1:], dtype=np.float32)
    n_nonzero_votes = np.count_nonzero(stack, axis=0).astype(np.int16)

    flat_stack = stack.reshape(stack.shape[0], -1)
    flat_consensus = consensus.reshape(-1)
    flat_fraction = max_vote_fraction.reshape(-1)
    for idx in range(flat_stack.shape[1]):
        labels = [int(value) for value in flat_stack[:, idx] if int(value) != 0]
        if len(labels) < min_votes:
            continue
        counts = Counter(labels)
        label, votes = counts.most_common(1)[0]
        if list(counts.values()).count(votes) > 1:
            continue
        flat_consensus[idx] = label
        flat_fraction[idx] = votes / len(arrays)
    return consensus, max_vote_fraction, n_nonzero_votes


def dice_jaccard(source: np.ndarray, reference: np.ndarray, label: int) -> tuple[float, float, int, int, int]:
    src = source == label
    ref = reference == label
    source_voxels = int(np.count_nonzero(src))
    reference_voxels = int(np.count_nonzero(ref))
    intersection = int(np.count_nonzero(src & ref))
    union = int(np.count_nonzero(src | ref))
    dice = 2.0 * intersection / (source_voxels + reference_voxels) if source_voxels + reference_voxels else math.nan
    jaccard = intersection / union if union else math.nan
    return dice, jaccard, source_voxels, reference_voxels, intersection


def pct_error(source_voxels: int, reference_voxels: int) -> float:
    if reference_voxels == 0:
        return math.nan
    return (source_voxels - reference_voxels) / reference_voxels * 100.0


def median(values: list[float]) -> float:
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return math.nan
    n = len(clean)
    mid = n // 2
    if n % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def percentile(values: list[float], pct: float) -> float:
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return math.nan
    return clean[round((len(clean) - 1) * pct)]


def compare_to_reference(
    source: Source,
    source_array: np.ndarray,
    reference_array: np.ndarray,
    variant: str,
    n_reference_sources: int,
    min_reference_voxels: int,
) -> list[dict[str, str | int | float]]:
    rows = []
    for label, structure in ASEG_LABELS.items():
        dice, jaccard, src_vox, ref_vox, inter = dice_jaccard(source_array, reference_array, label)
        if ref_vox < min_reference_voxels:
            continue
        signed_volume_error = pct_error(src_vox, ref_vox)
        rows.append(
            {
                "reference_variant": variant,
                "source_id": source.source_id,
                "method": source.method,
                "scan_id": source.scan_id,
                "structure": structure,
                "label": label,
                "dice": dice,
                "jaccard": jaccard,
                "source_voxels": src_vox,
                "reference_voxels": ref_vox,
                "intersection_voxels": inter,
                "signed_volume_error_pct": signed_volume_error,
                "abs_volume_error_pct": abs(signed_volume_error) if not math.isnan(signed_volume_error) else math.nan,
                "n_reference_sources": n_reference_sources,
            }
        )
    return rows


def summarize(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str | int | float]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["reference_variant"]),
                str(row["source_id"]),
                str(row["method"]),
                str(row["scan_id"]),
            )
        ].append(row)

    out = []
    for (variant, source_id, method, scan_id), group in sorted(grouped.items()):
        dice_values = [float(row["dice"]) for row in group]
        err_values = [float(row["abs_volume_error_pct"]) for row in group]
        out.append(
            {
                "reference_variant": variant,
                "source_id": source_id,
                "method": method,
                "scan_id": scan_id,
                "n_structures": len(group),
                "median_dice": median(dice_values),
                "p10_dice": percentile(dice_values, 0.10),
                "median_abs_volume_error_pct": median(err_values),
                "p90_abs_volume_error_pct": percentile(err_values, 0.90),
                "interpretation": interpret_dice(median(dice_values)),
            }
        )
    return out


def interpret_dice(value: float) -> str:
    if math.isnan(value):
        return "not_evaluable"
    if value >= 0.85:
        return "high_spatial_agreement"
    if value >= 0.70:
        return "moderate_spatial_agreement"
    if value >= 0.50:
        return "low_spatial_agreement"
    return "severe_spatial_disagreement"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--min-reference-voxels", type=int, default=50)
    args = parser.parse_args()

    sources = read_manifest(args.input_manifest, args.data_root)
    target_source, target_img = load_target(sources)
    arrays = {source.source_id: resample_label_map(source.path, target_img) for source in sources}

    trusted_sources = [source for source in sources if source.include_in_reference]
    trusted_arrays = [arrays[source.source_id] for source in trusted_sources]
    consensus, vote_fraction, n_nonzero_votes = build_consensus(trusted_arrays, min_votes=args.min_votes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(consensus, target_img.affine, target_img.header), str(args.output_dir / "pseudo_gt_trusted_hard_vote.nii.gz"))
    nib.save(nib.Nifti1Image(vote_fraction, target_img.affine, target_img.header), str(args.output_dir / "pseudo_gt_trusted_vote_fraction.nii.gz"))
    nib.save(nib.Nifti1Image(n_nonzero_votes, target_img.affine, target_img.header), str(args.output_dir / "pseudo_gt_trusted_vote_count.nii.gz"))

    metric_rows = []
    for source in sources:
        metric_rows.extend(
            compare_to_reference(
                source,
                arrays[source.source_id],
                consensus,
                "trusted_hard_vote_header_affine",
                len(trusted_sources),
                args.min_reference_voxels,
            )
        )
        if source.include_in_reference:
            loo_sources = [item for item in trusted_sources if item.source_id != source.source_id]
            if len(loo_sources) >= args.min_votes:
                loo_arrays = [arrays[item.source_id] for item in loo_sources]
                loo_consensus, _, _ = build_consensus(loo_arrays, min_votes=args.min_votes)
                metric_rows.extend(
                    compare_to_reference(
                        source,
                        arrays[source.source_id],
                        loo_consensus,
                        "trusted_leave_one_source_out_hard_vote_header_affine",
                        len(loo_sources),
                        args.min_reference_voxels,
                    )
                )

    summary_rows = summarize(metric_rows)
    write_csv(args.output_dir / "pseudo_gt_spatial_header_affine_metrics.csv", metric_rows)
    write_csv(args.output_dir / "pseudo_gt_spatial_header_affine_source_summary.csv", summary_rows)

    metadata = {
        "analysis": "header-affine spatial pseudo-GT pilot",
        "input_manifest": str(args.input_manifest),
        "data_root": str(args.data_root),
        "target_grid_source_id": target_source.source_id,
        "target_shape": list(target_img.shape),
        "min_votes": args.min_votes,
        "min_reference_voxels": args.min_reference_voxels,
        "n_sources": len(sources),
        "n_trusted_sources": len(trusted_sources),
        "n_metric_rows": len(metric_rows),
        "limitations": [
            "Uses only NIfTI header-affine resampling; no deformable registration.",
            "Hard-vote pseudo-GT is not anatomical truth.",
            "Surface distances are not computed in this pilot.",
        ],
    }
    (args.output_dir / "pseudo_gt_spatial_header_affine_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote spatial pseudo-GT pilot to: {args.output_dir}")


if __name__ == "__main__":
    main()
