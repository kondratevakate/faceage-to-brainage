"""Evaluate 3DDFA photo-ensemble stability against an MRI face target.

This script is a diagnostic layer above ``landmark_mask_overlay.py``. It keeps
all photos on one MRI-front-plane grid, then reports:

* each photo mask vs MRI target;
* each photo mask vs the other 3DDFA photo masks;
* consensus masks at multiple vote thresholds vs MRI target;
* leave-one-photo-out consensus stability.

The output is not a full-face anatomical accuracy claim. It is a front-plane,
landmark-constrained reproducibility check for one-photo avatar baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from landmark_mask_overlay import (
    SUPPORT_NAMES,
    apply_similarity_2d,
    extract_source_landmarks,
    load_3ddfa_keypoint_vertex_ids,
    load_geometry,
    mask_metrics,
    mesh_to_mask,
    maybe_swap_lr,
    mri_front_points,
    points_to_mask,
    raster_grid,
    similarity_2d,
    source_front_points,
    source_region_mask,
    vector_anchor_similarity,
)


def photo_label(path: Path) -> str:
    match = re.search(r"photo_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", path.name)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return path.stem


def read_target(
    mesh_path: Path,
    metadata_path: Path,
    nose_drop_mm: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    vertices, _faces = load_geometry(mesh_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    nose_z = float(metadata["landmarks"]["nose_tip"][2])
    target_vertices = vertices[vertices[:, 2] >= nose_z - nose_drop_mm]
    target_points_2d = mri_front_points(target_vertices)
    target_landmarks_2d = {
        name: np.asarray(metadata["landmarks"][name], dtype=np.float64)[[0, 2]]
        for name in SUPPORT_NAMES
    }
    return target_points_2d, target_landmarks_2d


def fit_photo(
    mesh_path: Path,
    target_landmarks_2d: dict[str, np.ndarray],
    keypoint_ids: np.ndarray,
    alignment_policy: str,
    source_fraction_below_nose: float,
    allow_lr_swap: bool,
) -> dict:
    vertices, faces = load_geometry(mesh_path)
    landmarks_3d = extract_source_landmarks(vertices, "3ddfa_v2", keypoint_ids)
    source_points_2d = source_front_points(vertices)
    candidates = [("normal", landmarks_3d)]
    if allow_lr_swap:
        candidates.append(("lr_swapped", maybe_swap_lr(landmarks_3d)))

    best = None
    dst_lm = np.vstack([target_landmarks_2d[name] for name in SUPPORT_NAMES])
    for orientation, candidate_landmarks in candidates:
        src_landmarks_2d = {
            name: source_front_points(candidate_landmarks[name].reshape(1, 3))[0]
            for name in SUPPORT_NAMES
        }
        src_lm = np.vstack([src_landmarks_2d[name] for name in SUPPORT_NAMES])
        if alignment_policy == "similarity_support":
            r, scale, t = similarity_2d(src_lm, dst_lm)
        elif alignment_policy == "support_similarity_nose_anchor":
            r, scale, _t = similarity_2d(src_lm, dst_lm)
            t = target_landmarks_2d["nose_tip"] - scale * (r @ src_landmarks_2d["nose_tip"])
        elif alignment_policy == "nose_cheek_axis":
            r, scale, t = vector_anchor_similarity(
                src_landmarks_2d,
                target_landmarks_2d,
                "left_cheek",
                "right_cheek",
            )
        elif alignment_policy == "nose_brow_axis":
            r, scale, t = vector_anchor_similarity(
                src_landmarks_2d,
                target_landmarks_2d,
                "nose_tip",
                "brow_center",
            )
        else:
            raise ValueError(alignment_policy)

        aligned_points = apply_similarity_2d(source_points_2d, r, scale, t)
        aligned_landmarks = {
            name: apply_similarity_2d(src_landmarks_2d[name].reshape(1, 2), r, scale, t)[0]
            for name in SUPPORT_NAMES
        }
        residuals = np.linalg.norm(
            np.vstack([aligned_landmarks[name] for name in SUPPORT_NAMES]) - dst_lm,
            axis=1,
        )
        vertex_keep = source_region_mask(vertices, candidate_landmarks, source_fraction_below_nose)
        row = {
            "mesh_path": str(mesh_path),
            "mesh_name": mesh_path.name,
            "photo_label": photo_label(mesh_path),
            "orientation": orientation,
            "scale_2d": float(scale),
            "landmark_rmse_mm": float(np.sqrt(np.mean(residuals**2))),
            "landmark_median_mm": float(np.median(residuals)),
            "landmark_max_mm": float(np.max(residuals)),
            "landmark_residuals_json": json.dumps(
                {name: float(value) for name, value in zip(SUPPORT_NAMES, residuals)}
            ),
            "aligned_points_2d": aligned_points,
            "aligned_landmarks_2d": aligned_landmarks,
            "faces": faces,
            "vertex_keep": vertex_keep,
            "source_vertices_in_mask": int(vertex_keep.sum()),
            "source_vertices_total": int(len(vertices)),
        }
        score = row["landmark_rmse_mm"]
        if best is None or score < best["landmark_rmse_mm"]:
            best = row
    if best is None:
        raise ValueError(f"No fit candidate for {mesh_path}")
    return best


def clean_metric_row(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key
        not in {
            "aligned_points_2d",
            "aligned_landmarks_2d",
            "faces",
            "vertex_keep",
            "source_mask",
        }
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def summarize_values(rows: list[dict], metrics: list[str]) -> dict:
    out = {"n": len(rows)}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        out[f"{metric}_min"] = float(np.min(values))
        out[f"{metric}_median"] = float(np.median(values))
        out[f"{metric}_max"] = float(np.max(values))
        out[f"{metric}_range"] = float(np.max(values) - np.min(values))
    return out


def consensus_mask(masks: np.ndarray, min_votes: int) -> np.ndarray:
    return masks.sum(axis=0) >= min_votes


def add_metric_prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def plot_ensemble(
    output: Path,
    grid: dict,
    target_mask: np.ndarray,
    masks: np.ndarray,
    consensus_rows: list[dict],
    per_photo_rows: list[dict],
    qc_rows: list[dict],
) -> None:
    votes = masks.sum(axis=0)
    n = masks.shape[0]
    consensus_2 = consensus_mask(masks, min_votes=max(1, int(np.ceil(n / 2))))
    consensus_3 = consensus_mask(masks, min_votes=min(n, 3))
    extent = grid["extent"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=150)
    ax = axes[0, 0]
    ax.imshow(target_mask, origin="lower", extent=extent, cmap="Blues", alpha=0.5, interpolation="nearest")
    ax.imshow(consensus_2, origin="lower", extent=extent, cmap="Reds", alpha=0.42, interpolation="nearest")
    row_2 = next(row for row in consensus_rows if row["min_votes"] == max(1, int(np.ceil(n / 2))))
    ax.set_title(f"Consensus >= {row_2['min_votes']}/{n} vs MRI\nDice {row_2['dice']:.3f}, HD95 {row_2['boundary_hd95_mm']:.1f} mm")
    ax.set_aspect("equal")

    ax = axes[0, 1]
    ax.imshow(target_mask, origin="lower", extent=extent, cmap="Blues", alpha=0.5, interpolation="nearest")
    ax.imshow(consensus_3, origin="lower", extent=extent, cmap="Reds", alpha=0.42, interpolation="nearest")
    row_3 = next((row for row in consensus_rows if row["min_votes"] == min(n, 3)), None)
    if row_3:
        ax.set_title(f"Stable core >= {row_3['min_votes']}/{n} vs MRI\nDice {row_3['dice']:.3f}, HD95 {row_3['boundary_hd95_mm']:.1f} mm")
    ax.set_aspect("equal")

    ax = axes[0, 2]
    im = ax.imshow(votes, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=n, interpolation="nearest")
    ax.contour(target_mask.astype(float), levels=[0.5], extent=extent, origin="lower", colors="#2563eb", linewidths=0.8)
    ax.set_title("3DDFA photo vote count")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = [row["photo_label"].split()[-1] for row in per_photo_rows]
    x = np.arange(len(labels))
    ax = axes[1, 0]
    ax.bar(x, [row["dice"] for row in per_photo_rows], color="#b91c1c", alpha=0.75)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice vs MRI")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.bar(x, [row["boundary_hd95_mm"] for row in per_photo_rows], color="#1d4ed8", alpha=0.75)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("HD95 vs MRI, mm")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 2]
    ax.bar(x, [row["dice_to_other_consensus"] for row in qc_rows], color="#047857", alpha=0.75)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice vs other-photo consensus")
    ax.grid(axis="y", alpha=0.25)

    for ax in axes.ravel()[:3]:
        ax.set_xlabel("MRI x, mm")
        ax.set_ylabel("MRI z, mm")
    fig.suptitle("3DDFA photo ensemble QC, front-plane landmark-constrained")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mri-patch-mesh", required=True, type=Path)
    parser.add_argument("--mri-patch-metadata", required=True, type=Path)
    parser.add_argument("--photo-mesh-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*1_1_photo*face1.ply")
    parser.add_argument("--bfm-pkl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--nose-drop-mm", type=float, default=35.0)
    parser.add_argument("--resolution-mm", type=float, default=1.0)
    parser.add_argument("--margin-mm", type=float, default=8.0)
    parser.add_argument("--target-dilation-iters", type=int, default=1)
    parser.add_argument("--source-dilation-iters", type=int, default=1)
    parser.add_argument("--source-fraction-below-nose", type=float, default=0.35)
    parser.add_argument("--allow-lr-swap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--alignment-policy",
        choices=["similarity_support", "support_similarity_nose_anchor", "nose_cheek_axis", "nose_brow_axis"],
        default="support_similarity_nose_anchor",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_points_2d, target_landmarks_2d = read_target(
        args.mri_patch_mesh,
        args.mri_patch_metadata,
        args.nose_drop_mm,
    )
    keypoint_ids = load_3ddfa_keypoint_vertex_ids(args.bfm_pkl)
    mesh_paths = sorted(args.photo_mesh_dir.glob(args.pattern))
    if len(mesh_paths) < 2:
        raise ValueError(f"Need at least two 3DDFA meshes, got {len(mesh_paths)}")

    photos = [
        fit_photo(
            mesh_path,
            target_landmarks_2d,
            keypoint_ids,
            args.alignment_policy,
            args.source_fraction_below_nose,
            args.allow_lr_swap,
        )
        for mesh_path in mesh_paths
    ]

    kept_points = [photo["aligned_points_2d"][photo["vertex_keep"]] for photo in photos]
    grid = raster_grid(target_points_2d, np.vstack(kept_points), args.resolution_mm, args.margin_mm)
    target_mask = points_to_mask(target_points_2d, grid, args.target_dilation_iters)
    for photo in photos:
        photo["source_mask"] = mesh_to_mask(
            photo["aligned_points_2d"],
            photo["faces"],
            photo["vertex_keep"],
            grid,
            args.source_dilation_iters,
        )

    per_photo_rows = []
    for index, photo in enumerate(photos, start=1):
        metrics = mask_metrics(photo["source_mask"], target_mask, args.resolution_mm)
        row = {
            "index": index,
            "method": "3ddfa_v2",
            "alignment_policy": args.alignment_policy,
            **clean_metric_row(photo),
            **metrics,
        }
        per_photo_rows.append(row)
    write_rows(args.output_dir / "per_photo_to_mri_metrics.csv", per_photo_rows)

    pairwise_rows = []
    for (i, left), (j, right) in combinations(enumerate(photos, start=1), 2):
        metrics = mask_metrics(left["source_mask"], right["source_mask"], args.resolution_mm)
        pairwise_rows.append(
            {
                "left_index": i,
                "right_index": j,
                "left_photo_label": left["photo_label"],
                "right_photo_label": right["photo_label"],
                "left_mesh_name": left["mesh_name"],
                "right_mesh_name": right["mesh_name"],
                **metrics,
            }
        )
    write_rows(args.output_dir / "pairwise_photo_mask_metrics.csv", pairwise_rows)

    masks = np.stack([photo["source_mask"] for photo in photos], axis=0)
    n = masks.shape[0]
    consensus_thresholds = sorted(set([1, max(1, int(np.ceil(n / 2))), min(n, 3), n]))
    consensus_rows = []
    for min_votes in consensus_thresholds:
        c_mask = consensus_mask(masks, min_votes)
        row = {
            "consensus_name": f"votes_ge_{min_votes}_of_{n}",
            "min_votes": min_votes,
            "n_photos": n,
            **mask_metrics(c_mask, target_mask, args.resolution_mm),
        }
        consensus_rows.append(row)
    write_rows(args.output_dir / "consensus_to_mri_metrics.csv", consensus_rows)

    leave_one_rows = []
    full_majority = consensus_mask(masks, max(1, int(np.ceil(n / 2))))
    for leave_index in range(n):
        other_masks = np.delete(masks, leave_index, axis=0)
        threshold = max(1, int(np.ceil(other_masks.shape[0] / 2)))
        loo_mask = consensus_mask(other_masks, threshold)
        row = {
            "left_out_index": leave_index + 1,
            "left_out_photo_label": photos[leave_index]["photo_label"],
            "other_n_photos": int(other_masks.shape[0]),
            "other_min_votes": int(threshold),
            **add_metric_prefix("loo_consensus_to_mri", mask_metrics(loo_mask, target_mask, args.resolution_mm)),
            **add_metric_prefix("loo_consensus_to_full_majority", mask_metrics(loo_mask, full_majority, args.resolution_mm)),
        }
        leave_one_rows.append(row)
    write_rows(args.output_dir / "leave_one_out_consensus_metrics.csv", leave_one_rows)

    qc_rows = []
    for index, photo in enumerate(photos, start=1):
        other_masks = np.delete(masks, index - 1, axis=0)
        threshold = max(1, int(np.ceil(other_masks.shape[0] / 2)))
        other_consensus = consensus_mask(other_masks, threshold)
        consistency = mask_metrics(photo["source_mask"], other_consensus, args.resolution_mm)
        pair_dice = [
            row["dice"]
            for row in pairwise_rows
            if row["left_index"] == index or row["right_index"] == index
        ]
        pair_hd95 = [
            row["boundary_hd95_mm"]
            for row in pairwise_rows
            if row["left_index"] == index or row["right_index"] == index
        ]
        qc_rows.append(
            {
                "index": index,
                "photo_label": photo["photo_label"],
                "mesh_name": photo["mesh_name"],
                "other_consensus_min_votes": threshold,
                "dice_to_other_consensus": consistency["dice"],
                "hd95_to_other_consensus_mm": consistency["boundary_hd95_mm"],
                "assd_to_other_consensus_mm": consistency["boundary_assd_mm"],
                "median_pairwise_dice": float(np.median(pair_dice)),
                "median_pairwise_hd95_mm": float(np.median(pair_hd95)),
                "scale_2d": photo["scale_2d"],
                "landmark_rmse_mm": photo["landmark_rmse_mm"],
                "source_area_mm2": per_photo_rows[index - 1]["source_area_mm2"],
                "mri_dice": per_photo_rows[index - 1]["dice"],
                "mri_hd95_mm": per_photo_rows[index - 1]["boundary_hd95_mm"],
            }
        )
    write_rows(args.output_dir / "avatar_only_photo_qc.csv", qc_rows)

    votes = masks.sum(axis=0)
    unstable = (votes > 0) & (votes < n)
    summary = {
        "method": "3ddfa_v2",
        "alignment_policy": args.alignment_policy,
        "mri_patch_mesh": str(args.mri_patch_mesh),
        "mri_patch_metadata": str(args.mri_patch_metadata),
        "photo_mesh_dir": str(args.photo_mesh_dir),
        "pattern": args.pattern,
        "n_photos": n,
        "photo_to_mri_summary": summarize_values(
            per_photo_rows,
            ["dice", "boundary_hd95_mm", "boundary_assd_mm", "landmark_rmse_mm", "scale_2d", "source_area_mm2"],
        ),
        "photo_to_photo_summary": summarize_values(
            pairwise_rows,
            ["dice", "boundary_hd95_mm", "boundary_assd_mm"],
        ),
        "photo_to_other_consensus_summary": summarize_values(
            qc_rows,
            ["dice_to_other_consensus", "hd95_to_other_consensus_mm", "median_pairwise_dice"],
        ),
        "consensus_to_mri_metrics": consensus_rows,
        "unstable_vote_area_mm2": float(unstable.sum() * args.resolution_mm * args.resolution_mm),
        "any_photo_vote_area_mm2": float((votes > 0).sum() * args.resolution_mm * args.resolution_mm),
        "all_photo_vote_area_mm2": float((votes == n).sum() * args.resolution_mm * args.resolution_mm),
        "outputs": {
            "per_photo_to_mri_csv": str(args.output_dir / "per_photo_to_mri_metrics.csv"),
            "pairwise_photo_mask_csv": str(args.output_dir / "pairwise_photo_mask_metrics.csv"),
            "consensus_to_mri_csv": str(args.output_dir / "consensus_to_mri_metrics.csv"),
            "leave_one_out_csv": str(args.output_dir / "leave_one_out_consensus_metrics.csv"),
            "avatar_only_photo_qc_csv": str(args.output_dir / "avatar_only_photo_qc.csv"),
            "ensemble_qc_png": str(args.output_dir / "3ddfa_ensemble_qc.png"),
        },
        "validity_note": "Front-plane, landmark-constrained diagnostic only. Avatar depth and full-face accuracy are not validated here.",
    }
    (args.output_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_ensemble(
        args.output_dir / "3ddfa_ensemble_qc.png",
        grid,
        target_mask,
        masks,
        consensus_rows,
        per_photo_rows,
        qc_rows,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
