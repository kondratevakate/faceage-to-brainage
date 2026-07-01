#!/usr/bin/env python3
"""Generate visual QC overlays for Kate n=1 segmentation outputs.

Runtime PNGs are written outside git under the local reprocessed data root.
Small CSV manifests can be copied into the repo for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reprocessed_2026" / "qc_overlays" / "kate_n1_2026"
DEFAULT_REGISTERED_SOURCES = REPO_ROOT / "data" / "kate_n1_2026" / "pseudo_gt_spatial_registered_sources.csv"
DEFAULT_REGISTERED_SUMMARY = REPO_ROOT / "data" / "kate_n1_2026" / "pseudo_gt_spatial_registered_source_summary.csv"
DEFAULT_MANIFEST_COPY = REPO_ROOT / "data" / "kate_n1_2026" / "visual_qc_overlay_manifest.csv"

TIGERBX_2024_SCAN_IDS = [
    "kate_2024_3di",
    "kate_2024_t1_ffe_401",
    "kate_2024_t1_ffe_601",
]
TIGERBX_OUTPUT_TYPES = ["tbetmask", "aseg", "dgm"]
PLANES = [
    ("axial", 0),
    ("coronal", 1),
    ("sagittal", 2),
]


@dataclass
class OverlayRecord:
    overlay_group: str
    source_id: str
    method: str
    scan_id: str
    output_type: str
    overlay_path: str
    n_nonzero_voxels: int
    n_labels: int
    label_min: int
    label_max: int
    notes: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[OverlayRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(OverlayRecord.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def load_array(path: Path, pixel_type: int | None = None) -> tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(str(path), pixel_type) if pixel_type is not None else sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    return array, image


def normalize_background(background: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = background[np.isfinite(background)]
    if mask is not None and np.any(mask):
        masked = background[mask & np.isfinite(background)]
        if masked.size:
            values = masked
    if values.size == 0:
        return np.zeros_like(background, dtype=np.float32)
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        hi = float(values.max())
        lo = float(values.min())
    if hi <= lo:
        return np.zeros_like(background, dtype=np.float32)
    clipped = np.clip(background, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def selected_slices(mask: np.ndarray, axis: int, n_slices: int = 5) -> list[int]:
    coords = np.where(mask)
    if not coords[axis].size:
        return [mask.shape[axis] // 2]
    unique = np.unique(coords[axis])
    if unique.size <= n_slices:
        return [int(idx) for idx in unique]
    quantiles = np.linspace(0.15, 0.85, n_slices)
    return sorted({int(round(np.quantile(unique, q))) for q in quantiles})


def take_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        plane = array[index, :, :]
    elif axis == 1:
        plane = array[:, index, :]
    elif axis == 2:
        plane = array[:, :, index]
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    return np.rot90(plane)


def label_boundary(label2d: np.ndarray) -> np.ndarray:
    mask = label2d != 0
    if not np.any(mask):
        return mask
    boundary = mask & (
        (np.roll(label2d, 1, axis=0) != label2d)
        | (np.roll(label2d, -1, axis=0) != label2d)
        | (np.roll(label2d, 1, axis=1) != label2d)
        | (np.roll(label2d, -1, axis=1) != label2d)
    )
    boundary[0, :] = mask[0, :]
    boundary[-1, :] = mask[-1, :]
    boundary[:, 0] = mask[:, 0]
    boundary[:, -1] = mask[:, -1]
    return boundary


def label_rgba(label2d: np.ndarray, alpha: float) -> np.ndarray:
    cmap = plt.get_cmap("tab20")
    rgba = cmap((label2d.astype(np.int32) % 20) / 19.0)
    rgba[..., 3] = (label2d != 0) * alpha
    return rgba


def boundary_rgba(mask2d: np.ndarray, color: tuple[float, float, float], alpha: float = 0.9) -> np.ndarray:
    rgba = np.zeros(mask2d.shape + (4,), dtype=np.float32)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = mask2d.astype(np.float32) * alpha
    return rgba


def render_montage(
    *,
    background: np.ndarray,
    source_label: np.ndarray,
    output_path: Path,
    title: str,
    reference_label: np.ndarray | None = None,
    source_alpha: float = 0.26,
) -> None:
    mask = source_label != 0
    if reference_label is not None:
        mask = mask | (reference_label != 0)
    background_norm = normalize_background(background, mask=mask)

    n_cols = 5
    fig, axes = plt.subplots(len(PLANES), n_cols, figsize=(15, 8.2), dpi=150)
    fig.suptitle(title, fontsize=12)
    for row_idx, (plane_name, axis) in enumerate(PLANES):
        indices = selected_slices(mask, axis, n_slices=n_cols)
        while len(indices) < n_cols:
            indices.append(indices[-1])
        for col_idx, index in enumerate(indices[:n_cols]):
            ax = axes[row_idx, col_idx]
            bg = take_slice(background_norm, axis, index)
            src = take_slice(source_label, axis, index)
            ax.imshow(bg, cmap="gray", interpolation="nearest")
            ax.imshow(label_rgba(src, alpha=source_alpha), interpolation="nearest")
            ax.imshow(boundary_rgba(label_boundary(src), (1.0, 0.2, 0.05), alpha=0.95), interpolation="nearest")
            if reference_label is not None:
                ref = take_slice(reference_label, axis, index)
                ax.imshow(boundary_rgba(label_boundary(ref), (0.0, 0.85, 1.0), alpha=0.9), interpolation="nearest")
            ax.set_title(f"{plane_name} {index}", fontsize=7)
            ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def label_stats(label: np.ndarray) -> tuple[int, int, int, int]:
    nonzero = label[label != 0]
    if nonzero.size == 0:
        return 0, 0, 0, 0
    unique = np.unique(nonzero)
    return int(nonzero.size), int(unique.size), int(unique.min()), int(unique.max())


def generate_tigerbx_native(data_root: Path, output_root: Path) -> list[OverlayRecord]:
    tigerbx_root = data_root / "reprocessed_2026" / "asian_mri_tools" / "tigerbx" / "bx"
    rows: list[OverlayRecord] = []
    for scan_id in TIGERBX_2024_SCAN_IDS:
        background_path = tigerbx_root / f"{scan_id}_tbet.nii.gz"
        background, _ = load_array(background_path, sitk.sitkFloat32)
        for output_type in TIGERBX_OUTPUT_TYPES:
            label_path = tigerbx_root / f"{scan_id}_{output_type}.nii.gz"
            label, _ = load_array(label_path, sitk.sitkUInt16)
            overlay_path = output_root / "tigerbx_native" / f"{scan_id}_{output_type}_overlay.png"
            render_montage(
                background=background,
                source_label=label.astype(np.int16),
                output_path=overlay_path,
                title=f"TIGERBx native {scan_id} {output_type}: fill/contour = source labels",
                reference_label=None,
                source_alpha=0.30 if output_type == "tbetmask" else 0.24,
            )
            n_voxels, n_labels, label_min, label_max = label_stats(label)
            rows.append(
                OverlayRecord(
                    overlay_group="tigerbx_native",
                    source_id=f"tigerbx_{scan_id}_{output_type}",
                    method="TIGERBx",
                    scan_id=scan_id,
                    output_type=output_type,
                    overlay_path=str(overlay_path),
                    n_nonzero_voxels=n_voxels,
                    n_labels=n_labels,
                    label_min=label_min,
                    label_max=label_max,
                    notes="Native TIGERBx output over TIGERBx brain-extracted image.",
                )
            )
    return rows


def resample_fixed_image_to_label_grid(fixed_image_path: Path, reference_label_path: Path) -> np.ndarray:
    fixed = sitk.ReadImage(str(fixed_image_path), sitk.sitkFloat32)
    reference = sitk.ReadImage(str(reference_label_path), sitk.sitkUInt16)
    resampled = sitk.Resample(
        fixed,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    return sitk.GetArrayFromImage(resampled)


def source_summary_lookup(path: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        if row.get("reference_variant") == "trusted_registered_hard_vote":
            lookup[row["source_id"]] = row
    return lookup


def generate_registered_pseudo_gt(
    data_root: Path,
    output_root: Path,
    sources_csv: Path,
    summary_csv: Path,
) -> list[OverlayRecord]:
    rows: list[OverlayRecord] = []
    sources = read_csv(sources_csv)
    fixed_candidates = [row for row in sources if row["source_id"] == "synthseg_2024_t1ffe_ax"]
    if len(fixed_candidates) != 1:
        raise ValueError("Expected synthseg_2024_t1ffe_ax as fixed image row.")
    fixed_image_path = Path(fixed_candidates[0]["image_path"])
    pseudo_gt_root = data_root / "reprocessed_2026" / "pseudo_gt" / "spatial_registered_v1"
    consensus_path = pseudo_gt_root / "pseudo_gt_registered_trusted_hard_vote.nii.gz"
    consensus, _ = load_array(consensus_path, sitk.sitkUInt16)
    background = resample_fixed_image_to_label_grid(fixed_image_path, consensus_path)
    summary = source_summary_lookup(summary_csv)

    for source in sources:
        source_id = source["source_id"]
        label_path = Path(source["registered_label_path"])
        label, _ = load_array(label_path, sitk.sitkUInt16)
        metric_note = ""
        if source_id in summary:
            metric_note = (
                f"median_dice={float(summary[source_id]['median_dice']):.3f}; "
                f"p90_hd95_mm={float(summary[source_id]['p90_hd95_mm']):.2f}; "
                f"median_abs_volume_error_pct={float(summary[source_id]['median_abs_volume_error_pct']):.2f}"
            )
        overlay_path = output_root / "registered_pseudo_gt" / f"{source_id}_vs_consensus_overlay.png"
        render_montage(
            background=background,
            source_label=label.astype(np.int16),
            reference_label=consensus.astype(np.int16),
            output_path=overlay_path,
            title=f"{source_id} vs trusted registered hard vote; orange=source, cyan=consensus",
            source_alpha=0.22,
        )
        n_voxels, n_labels, label_min, label_max = label_stats(label)
        rows.append(
            OverlayRecord(
                overlay_group="registered_pseudo_gt",
                source_id=source_id,
                method=source["method"],
                scan_id=source["scan_id"],
                output_type="registered_aseg_vs_consensus",
                overlay_path=str(overlay_path),
                n_nonzero_voxels=n_voxels,
                n_labels=n_labels,
                label_min=label_min,
                label_max=label_max,
                notes=metric_note,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual QC overlays for Kate n=1 segmentation branches.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--registered-sources", type=Path, default=DEFAULT_REGISTERED_SOURCES)
    parser.add_argument("--registered-summary", type=Path, default=DEFAULT_REGISTERED_SUMMARY)
    parser.add_argument("--manifest-copy", type=Path, default=DEFAULT_MANIFEST_COPY)
    parser.add_argument("--mode", choices=["all", "tigerbx", "registered"], default="all")
    args = parser.parse_args()

    records: list[OverlayRecord] = []
    if args.mode in {"all", "tigerbx"}:
        records.extend(generate_tigerbx_native(args.data_root, args.output_root))
    if args.mode in {"all", "registered"}:
        records.extend(
            generate_registered_pseudo_gt(
                args.data_root,
                args.output_root,
                args.registered_sources,
                args.registered_summary,
            )
        )

    runtime_manifest = args.output_root / "visual_qc_overlay_manifest.csv"
    write_manifest(runtime_manifest, records)
    if args.manifest_copy:
        write_manifest(args.manifest_copy, records)
    print(f"Wrote {len(records)} overlay records")
    print(f"Runtime manifest: {runtime_manifest}")
    if args.manifest_copy:
        print(f"Repo manifest copy: {args.manifest_copy}")


if __name__ == "__main__":
    main()
