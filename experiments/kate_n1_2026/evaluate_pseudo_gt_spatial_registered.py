#!/usr/bin/env python3
"""Registered spatial pseudo-GT evaluation for 2024 label maps.

This stage performs intensity-based affine registration to a fixed 2024 T1 FFE
image, resamples label maps into the fixed label grid, builds trusted hard-vote
pseudo-GT, and scores source segmentations with Dice/Jaccard/HD95/ASSD.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


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
    image_path: Path
    label_path: Path
    include_in_reference: bool
    reference_grid: bool
    notes: str


def resolve_path(data_root: Path, value: str) -> Path:
    path = data_root / value
    return path.resolve()


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
                    image_path=resolve_path(data_root, row["image_path"]),
                    label_path=resolve_path(data_root, row["label_path"]),
                    include_in_reference=row.get("include_in_reference", "0") == "1",
                    reference_grid=row.get("reference_grid", "0") == "1",
                    notes=row.get("notes", ""),
                )
            )
    return sources


def read_float_image(path: Path) -> sitk.Image:
    image = sitk.ReadImage(str(path), sitk.sitkFloat32)
    return sitk.Normalize(sitk.Cast(image, sitk.sitkFloat32))


def read_label_image(path: Path) -> sitk.Image:
    return sitk.Cast(sitk.ReadImage(str(path)), sitk.sitkUInt16)


def load_reference(sources: list[Source]) -> tuple[Source, sitk.Image, sitk.Image]:
    refs = [source for source in sources if source.reference_grid]
    if len(refs) != 1:
        raise ValueError(f"Expected one reference_grid=1 source, got {len(refs)}")
    source = refs[0]
    return source, read_float_image(source.image_path), read_label_image(source.label_path)


def register_affine(
    fixed: sitk.Image,
    moving: sitk.Image,
    output_tfm: Path,
) -> sitk.Transform:
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.05, seed=17)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=160,
        gradientMagnitudeTolerance=1e-6,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 1])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=False)
    transform = registration.Execute(fixed, moving)
    output_tfm.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(transform, str(output_tfm))
    return transform


def get_transform(
    source: Source,
    fixed_source: Source,
    fixed_image: sitk.Image,
    output_dir: Path,
) -> sitk.Transform:
    if source.image_path == fixed_source.image_path:
        return sitk.Transform(3, sitk.sitkIdentity)
    transform_path = output_dir / "transforms" / f"{source.scan_id}_to_{fixed_source.scan_id}_affine.tfm"
    if transform_path.exists():
        return sitk.ReadTransform(str(transform_path))
    moving = read_float_image(source.image_path)
    return register_affine(fixed_image, moving, transform_path)


def resample_label_to_reference(
    label: sitk.Image,
    reference_grid: sitk.Image,
    transform: sitk.Transform,
) -> sitk.Image:
    return sitk.Resample(
        label,
        reference_grid,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt16,
    )


def sitk_to_array(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image).astype(np.int16)


def array_to_label_image(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array.astype(np.uint16))
    image.CopyInformation(reference)
    return image


def build_consensus(arrays: list[np.ndarray], min_votes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack(arrays, axis=0)
    consensus = np.zeros(stack.shape[1:], dtype=np.uint16)
    max_vote_fraction = np.zeros(stack.shape[1:], dtype=np.float32)
    n_nonzero_votes = np.count_nonzero(stack, axis=0).astype(np.uint16)
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


def binary_image(array: np.ndarray, label: int, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray((array == label).astype(np.uint8))
    image.CopyInformation(reference)
    return image


def overlap_metrics(source: np.ndarray, reference: np.ndarray, label: int) -> tuple[float, float, int, int, int]:
    src = source == label
    ref = reference == label
    src_n = int(np.count_nonzero(src))
    ref_n = int(np.count_nonzero(ref))
    inter = int(np.count_nonzero(src & ref))
    union = int(np.count_nonzero(src | ref))
    dice = 2.0 * inter / (src_n + ref_n) if src_n + ref_n else math.nan
    jaccard = inter / union if union else math.nan
    return dice, jaccard, src_n, ref_n, inter


def surface_metrics(source_bin: sitk.Image, reference_bin: sitk.Image) -> tuple[float, float]:
    if int(sitk.GetArrayFromImage(source_bin).sum()) == 0 or int(sitk.GetArrayFromImage(reference_bin).sum()) == 0:
        return math.nan, math.nan
    source_surface = sitk.LabelContour(source_bin)
    reference_surface = sitk.LabelContour(reference_bin)
    source_surface_arr = sitk.GetArrayFromImage(source_surface).astype(bool)
    reference_surface_arr = sitk.GetArrayFromImage(reference_surface).astype(bool)
    if not source_surface_arr.any() or not reference_surface_arr.any():
        return math.nan, math.nan
    distance_to_reference = sitk.Abs(
        sitk.SignedMaurerDistanceMap(reference_surface, squaredDistance=False, useImageSpacing=True)
    )
    distance_to_source = sitk.Abs(
        sitk.SignedMaurerDistanceMap(source_surface, squaredDistance=False, useImageSpacing=True)
    )
    d_source_to_reference = sitk.GetArrayFromImage(distance_to_reference)[source_surface_arr]
    d_reference_to_source = sitk.GetArrayFromImage(distance_to_source)[reference_surface_arr]
    distances = np.concatenate([d_source_to_reference, d_reference_to_source])
    if distances.size == 0:
        return math.nan, math.nan
    return float(np.percentile(distances, 95)), float(np.mean(distances))


def pct_error(source_voxels: int, reference_voxels: int) -> float:
    if reference_voxels == 0:
        return math.nan
    return (source_voxels - reference_voxels) / reference_voxels * 100.0


def compare_to_reference(
    source: Source,
    source_array: np.ndarray,
    reference_array: np.ndarray,
    reference_grid: sitk.Image,
    variant: str,
    n_reference_sources: int,
    min_reference_voxels: int,
) -> list[dict[str, str | int | float]]:
    rows = []
    for label, structure in ASEG_LABELS.items():
        dice, jaccard, src_vox, ref_vox, inter = overlap_metrics(source_array, reference_array, label)
        if ref_vox < min_reference_voxels:
            continue
        source_bin = binary_image(source_array, label, reference_grid)
        reference_bin = binary_image(reference_array, label, reference_grid)
        hd95, assd = surface_metrics(source_bin, reference_bin)
        signed_error = pct_error(src_vox, ref_vox)
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
                "hd95_mm": hd95,
                "assd_mm": assd,
                "source_voxels": src_vox,
                "reference_voxels": ref_vox,
                "intersection_voxels": inter,
                "signed_volume_error_pct": signed_error,
                "abs_volume_error_pct": abs(signed_error) if not math.isnan(signed_error) else math.nan,
                "n_reference_sources": n_reference_sources,
            }
        )
    return rows


def median(values: list[float]) -> float:
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return math.nan
    n = len(clean)
    mid = n // 2
    return clean[mid] if n % 2 else (clean[mid - 1] + clean[mid]) / 2.0


def percentile(values: list[float], pct: float) -> float:
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return math.nan
    return clean[round((len(clean) - 1) * pct)]


def summarize(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str | int | float]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["reference_variant"]), str(row["source_id"]), str(row["method"]), str(row["scan_id"]))].append(row)
    summaries = []
    for (variant, source_id, method, scan_id), group in sorted(grouped.items()):
        dice_values = [float(row["dice"]) for row in group]
        hd95_values = [float(row["hd95_mm"]) for row in group]
        assd_values = [float(row["assd_mm"]) for row in group]
        error_values = [float(row["abs_volume_error_pct"]) for row in group]
        summaries.append(
            {
                "reference_variant": variant,
                "source_id": source_id,
                "method": method,
                "scan_id": scan_id,
                "n_structures": len(group),
                "median_dice": median(dice_values),
                "p10_dice": percentile(dice_values, 0.10),
                "median_hd95_mm": median(hd95_values),
                "p90_hd95_mm": percentile(hd95_values, 0.90),
                "median_assd_mm": median(assd_values),
                "median_abs_volume_error_pct": median(error_values),
                "p90_abs_volume_error_pct": percentile(error_values, 0.90),
                "interpretation": interpret_dice(median(dice_values)),
            }
        )
    return summaries


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
    fixed_source, fixed_image, reference_label_grid = load_reference(sources)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    resampled_arrays: dict[str, np.ndarray] = {}
    transform_rows = []
    for source in sources:
        transform = get_transform(source, fixed_source, fixed_image, args.output_dir)
        source_label = read_label_image(source.label_path)
        resampled_label = resample_label_to_reference(source_label, reference_label_grid, transform)
        resampled_arrays[source.source_id] = sitk_to_array(resampled_label)
        out_label = args.output_dir / "registered_labels" / f"{source.source_id}_registered_to_{fixed_source.scan_id}.nii.gz"
        out_label.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(resampled_label, str(out_label))
        transform_rows.append(
            {
                "source_id": source.source_id,
                "method": source.method,
                "scan_id": source.scan_id,
                "image_path": str(source.image_path),
                "label_path": str(source.label_path),
                "registered_label_path": str(out_label),
                "used_identity_transform": str(source.image_path == fixed_source.image_path).lower(),
            }
        )

    trusted_sources = [source for source in sources if source.include_in_reference]
    trusted_arrays = [resampled_arrays[source.source_id] for source in trusted_sources]
    consensus, vote_fraction, vote_count = build_consensus(trusted_arrays, min_votes=args.min_votes)
    sitk.WriteImage(array_to_label_image(consensus, reference_label_grid), str(args.output_dir / "pseudo_gt_registered_trusted_hard_vote.nii.gz"))
    vote_fraction_img = sitk.Cast(sitk.GetImageFromArray(vote_fraction), sitk.sitkFloat32)
    vote_fraction_img.CopyInformation(reference_label_grid)
    sitk.WriteImage(vote_fraction_img, str(args.output_dir / "pseudo_gt_registered_trusted_vote_fraction.nii.gz"))
    sitk.WriteImage(array_to_label_image(vote_count, reference_label_grid), str(args.output_dir / "pseudo_gt_registered_trusted_vote_count.nii.gz"))

    metric_rows = []
    for source in sources:
        metric_rows.extend(
            compare_to_reference(
                source,
                resampled_arrays[source.source_id],
                consensus,
                reference_label_grid,
                "trusted_registered_hard_vote",
                len(trusted_sources),
                args.min_reference_voxels,
            )
        )
        if source.include_in_reference:
            loo_sources = [item for item in trusted_sources if item.source_id != source.source_id]
            if len(loo_sources) >= args.min_votes:
                loo_arrays = [resampled_arrays[item.source_id] for item in loo_sources]
                loo_consensus, _, _ = build_consensus(loo_arrays, min_votes=args.min_votes)
                metric_rows.extend(
                    compare_to_reference(
                        source,
                        resampled_arrays[source.source_id],
                        loo_consensus,
                        reference_label_grid,
                        "trusted_leave_one_source_out_registered_hard_vote",
                        len(loo_sources),
                        args.min_reference_voxels,
                    )
                )

    summary_rows = summarize(metric_rows)
    write_csv(args.output_dir / "pseudo_gt_spatial_registered_metrics.csv", metric_rows)
    write_csv(args.output_dir / "pseudo_gt_spatial_registered_source_summary.csv", summary_rows)
    write_csv(args.output_dir / "pseudo_gt_spatial_registered_sources.csv", transform_rows)
    metadata = {
        "analysis": "registered spatial pseudo-GT evaluation",
        "input_manifest": str(args.input_manifest),
        "data_root": str(args.data_root),
        "fixed_source_id": fixed_source.source_id,
        "fixed_scan_id": fixed_source.scan_id,
        "fixed_image": str(fixed_source.image_path),
        "reference_label_grid": str(fixed_source.label_path),
        "min_votes": args.min_votes,
        "min_reference_voxels": args.min_reference_voxels,
        "n_sources": len(sources),
        "n_trusted_sources": len(trusted_sources),
        "n_metric_rows": len(metric_rows),
        "registration": "SimpleITK affine Mattes mutual information, multi-resolution",
        "limitations": [
            "Affine registration only; no deformable subject-template registration.",
            "Hard-vote pseudo-GT is not anatomical ground truth.",
            "Surface metrics are measured after affine registration in fixed label grid.",
        ],
    }
    (args.output_dir / "pseudo_gt_spatial_registered_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote registered spatial pseudo-GT evaluation to: {args.output_dir}")


if __name__ == "__main__":
    main()
