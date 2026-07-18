"""Overlay MRI and avatar face masks by support landmarks.

The mask overlay is determined by landmarks only. It does not select a visual
fit by eye. The current use case is an internal diagnostic for one-photo avatar
baselines against the measurable MRI front face patch.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import trimesh
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
)
from skimage.draw import polygon


SUPPORT_NAMES = ["nose_tip", "brow_center", "left_cheek", "right_cheek"]

MEDIAPIPE_LANDMARKS = {
    "nose_tip": [1],
    "brow_center": [168],
    "left_cheek": [234],
    "right_cheek": [454],
    "chin": [152],
}

TDDFA_68_LANDMARKS = {
    "nose_tip": [30],
    "brow_center": [27],
    "left_cheek": [2],
    "right_cheek": [14],
    "chin": [8],
}


def load_geometry(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        geom = trimesh.util.concatenate(tuple(geom.geometry.values()))
    if not hasattr(geom, "vertices"):
        raise ValueError(f"Cannot read vertices from {path}")
    vertices = np.asarray(geom.vertices, dtype=np.float64)
    vertices = vertices[np.isfinite(vertices).all(axis=1)]
    faces = None
    if hasattr(geom, "faces") and len(geom.faces):
        faces = np.asarray(geom.faces, dtype=np.int64)
    return vertices, faces


def load_3ddfa_keypoint_vertex_ids(bfm_pkl: Path) -> np.ndarray:
    with bfm_pkl.open("rb") as f:
        bfm = pickle.load(f)
    keypoints = np.asarray(bfm["keypoints"], dtype=np.int64)
    if len(keypoints) != 204:
        raise ValueError(f"Expected 204 flattened 68-landmark entries, got {len(keypoints)}")
    return keypoints[0::3] // 3


def extract_source_landmarks(
    vertices: np.ndarray,
    method: str,
    tddfa_keypoint_ids: np.ndarray | None,
) -> dict[str, np.ndarray]:
    if method == "mediapipe":
        mapping = MEDIAPIPE_LANDMARKS
        dense_vertices = vertices
    elif method == "3ddfa_v2":
        if tddfa_keypoint_ids is None:
            raise ValueError("--bfm-pkl is required for 3ddfa_v2")
        mapping = TDDFA_68_LANDMARKS
        dense_vertices = vertices[tddfa_keypoint_ids]
    else:
        raise ValueError(method)

    return {
        name: dense_vertices[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        for name, indices in mapping.items()
    }


def maybe_swap_lr(landmarks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    swapped = dict(landmarks)
    swapped["left_cheek"], swapped["right_cheek"] = landmarks["right_cheek"], landmarks["left_cheek"]
    return swapped


def similarity_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = src_c.T @ dst_c / len(src)
    u, s, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    scale = float(np.sum(s) / max(np.mean(np.sum(src_c * src_c, axis=1)), 1e-12))
    t = dst_mean - scale * (r @ src_mean)
    return r, scale, t


def vector_anchor_similarity(
    source_landmarks_2d: dict[str, np.ndarray],
    target_landmarks_2d: dict[str, np.ndarray],
    axis_from: str,
    axis_to: str,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Similarity transform with nose_tip as an exact translation anchor.

    Rotation and scale are estimated from one anatomical vector, while
    translation is set so the transformed source nose_tip lands exactly on the
    MRI nose_tip. This avoids a least-squares transform moving the nose in order
    to satisfy noisy cheek/brow proxies.
    """
    source_vec = source_landmarks_2d[axis_to] - source_landmarks_2d[axis_from]
    target_vec = target_landmarks_2d[axis_to] - target_landmarks_2d[axis_from]
    source_norm = float(np.linalg.norm(source_vec))
    target_norm = float(np.linalg.norm(target_vec))
    if source_norm < 1e-8 or target_norm < 1e-8:
        raise ValueError(f"Degenerate anchor vector: {axis_from}->{axis_to}")
    source_angle = float(np.arctan2(source_vec[1], source_vec[0]))
    target_angle = float(np.arctan2(target_vec[1], target_vec[0]))
    angle = target_angle - source_angle
    r = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )
    scale = target_norm / source_norm
    t = target_landmarks_2d["nose_tip"] - scale * (r @ source_landmarks_2d["nose_tip"])
    return r, float(scale), t


def apply_similarity_2d(points: np.ndarray, r: np.ndarray, scale: float, t: np.ndarray) -> np.ndarray:
    return (scale * (r @ points.T)).T + t


def source_front_points(vertices: np.ndarray) -> np.ndarray:
    # Current baselines store image-horizontal in x and image-vertical in y.
    return vertices[:, [0, 1]]


def mri_front_points(vertices: np.ndarray) -> np.ndarray:
    # MRI world front plane: left-right x and inferior-superior z.
    return vertices[:, [0, 2]]


def source_region_mask(vertices: np.ndarray, landmarks: dict[str, np.ndarray], fraction_below_nose: float) -> np.ndarray:
    nose = landmarks["nose_tip"]
    brow = landmarks["brow_center"]
    chin = landmarks["chin"]
    left = landmarks["left_cheek"]
    right = landmarks["right_cheek"]
    face_height = abs(float(brow[1] - chin[1]))
    cheek_width = abs(float(right[0] - left[0]))
    y_min = float(nose[1] - fraction_below_nose * face_height)
    x_min = min(float(left[0]), float(right[0])) - 0.12 * cheek_width
    x_max = max(float(left[0]), float(right[0])) + 0.12 * cheek_width
    keep = (vertices[:, 1] >= y_min) & (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max)
    if keep.sum() < 50:
        keep = vertices[:, 1] >= y_min
    return keep


def raster_grid(points_a: np.ndarray, points_b: np.ndarray, resolution_mm: float, margin_mm: float) -> dict:
    all_points = np.vstack([points_a, points_b])
    mins = all_points.min(axis=0) - margin_mm
    maxs = all_points.max(axis=0) + margin_mm
    width = int(np.ceil((maxs[0] - mins[0]) / resolution_mm)) + 1
    height = int(np.ceil((maxs[1] - mins[1]) / resolution_mm)) + 1
    return {
        "mins": mins,
        "maxs": maxs,
        "resolution_mm": resolution_mm,
        "shape": (height, width),
        "extent": [float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1])],
    }


def to_rc(points: np.ndarray, grid: dict) -> tuple[np.ndarray, np.ndarray]:
    mins = grid["mins"]
    res = grid["resolution_mm"]
    cols = np.round((points[:, 0] - mins[0]) / res).astype(np.int64)
    rows = np.round((points[:, 1] - mins[1]) / res).astype(np.int64)
    rows = np.clip(rows, 0, grid["shape"][0] - 1)
    cols = np.clip(cols, 0, grid["shape"][1] - 1)
    return rows, cols


def points_to_mask(points: np.ndarray, grid: dict, dilation_iters: int) -> np.ndarray:
    mask = np.zeros(grid["shape"], dtype=bool)
    rows, cols = to_rc(points, grid)
    mask[rows, cols] = True
    mask = binary_dilation(mask, iterations=dilation_iters)
    mask = binary_closing(mask, iterations=max(dilation_iters, 1))
    mask = binary_fill_holes(mask)
    return mask


def mesh_to_mask(
    points: np.ndarray,
    faces: np.ndarray | None,
    vertex_keep: np.ndarray,
    grid: dict,
    dilation_iters: int,
) -> np.ndarray:
    if faces is None:
        return points_to_mask(points[vertex_keep], grid, dilation_iters)

    mask = np.zeros(grid["shape"], dtype=bool)
    face_keep = vertex_keep[faces].all(axis=1)
    kept_faces = faces[face_keep]
    if len(kept_faces) == 0:
        return points_to_mask(points[vertex_keep], grid, dilation_iters)
    rows_all, cols_all = to_rc(points, grid)
    for face in kept_faces:
        rr, cc = polygon(rows_all[face], cols_all[face], shape=grid["shape"])
        mask[rr, cc] = True
    mask = binary_closing(mask, iterations=max(dilation_iters, 1))
    mask = binary_fill_holes(mask)
    return mask


def boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.copy()
    return mask ^ binary_erosion(mask)


def mask_metrics(source: np.ndarray, target: np.ndarray, resolution_mm: float) -> dict[str, float]:
    intersection = np.logical_and(source, target).sum()
    union = np.logical_or(source, target).sum()
    source_area = source.sum() * resolution_mm * resolution_mm
    target_area = target.sum() * resolution_mm * resolution_mm
    dice = 2 * intersection / max(source.sum() + target.sum(), 1)
    iou = intersection / max(union, 1)

    source_boundary = boundary(source)
    target_boundary = boundary(target)
    dist_to_target = distance_transform_edt(~target_boundary) * resolution_mm
    dist_to_source = distance_transform_edt(~source_boundary) * resolution_mm
    s2t = dist_to_target[source_boundary]
    t2s = dist_to_source[target_boundary]
    if len(s2t) == 0:
        s2t = np.array([np.nan])
    if len(t2s) == 0:
        t2s = np.array([np.nan])
    return {
        "source_area_mm2": float(source_area),
        "target_area_mm2": float(target_area),
        "dice": float(dice),
        "iou": float(iou),
        "boundary_s2t_median_mm": float(np.nanmedian(s2t)),
        "boundary_s2t_p95_mm": float(np.nanpercentile(s2t, 95)),
        "boundary_t2s_median_mm": float(np.nanmedian(t2s)),
        "boundary_t2s_p95_mm": float(np.nanpercentile(t2s, 95)),
        "boundary_hd95_mm": float(max(np.nanpercentile(s2t, 95), np.nanpercentile(t2s, 95))),
        "boundary_assd_mm": float((np.nanmean(s2t) + np.nanmean(t2s)) / 2),
    }


def write_overlay(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    grid: dict,
    source_landmarks_2d: dict[str, np.ndarray],
    target_landmarks_2d: dict[str, np.ndarray],
    output: Path,
    title: str,
    metrics: dict,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
    extent = grid["extent"]
    ax.imshow(target_mask, origin="lower", extent=extent, cmap="Blues", alpha=0.48, interpolation="nearest")
    ax.imshow(source_mask, origin="lower", extent=extent, cmap="Reds", alpha=0.42, interpolation="nearest")
    for name, point in target_landmarks_2d.items():
        ax.scatter([point[0]], [point[1]], s=34, c="#0b3d91", edgecolors="white", linewidths=0.5, zorder=5)
        ax.text(point[0] + 1, point[1] + 1, name, fontsize=6, color="#0b3d91")
    for name, point in source_landmarks_2d.items():
        ax.scatter([point[0]], [point[1]], s=30, c="#ff8c00", edgecolors="white", linewidths=0.5, zorder=6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("MRI x, mm")
    ax.set_ylabel("MRI z, mm")
    ax.grid(alpha=0.18, linewidth=0.5)
    x0 = extent[0] + 0.08 * (extent[1] - extent[0])
    y0 = extent[2] + 0.08 * (extent[3] - extent[2])
    ax.plot([x0, x0 + 20], [y0, y0], c="black", lw=2)
    ax.text(x0, y0 + 2, "20 mm", fontsize=8)
    handles = [
        Patch(facecolor="#2f80ed", alpha=0.48, label="MRI mask"),
        Patch(facecolor="#d62728", alpha=0.42, label="Avatar mask"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#0b3d91", markeredgecolor="white", markersize=7, label="MRI landmarks"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ff8c00", markeredgecolor="white", markersize=7, label="Avatar landmarks"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9, fontsize=8)
    text = (
        f"Dice={metrics['dice']:.3f}\n"
        f"IoU={metrics['iou']:.3f}\n"
        f"LM RMSE={metrics['landmark_rmse_mm']:.1f} mm\n"
        f"HD95={metrics['boundary_hd95_mm']:.1f} mm"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def evaluate_one(
    mesh_path: Path,
    method: str,
    target_points_2d: np.ndarray,
    target_landmarks_2d: dict[str, np.ndarray],
    args: argparse.Namespace,
    tddfa_keypoint_ids: np.ndarray | None,
) -> dict:
    vertices, faces = load_geometry(mesh_path)
    source_landmarks_3d = extract_source_landmarks(vertices, method, tddfa_keypoint_ids)
    source_points_2d = source_front_points(vertices)

    candidates = [("normal", source_landmarks_3d)]
    if args.allow_lr_swap:
        candidates.append(("lr_swapped", maybe_swap_lr(source_landmarks_3d)))

    best = None
    for orientation, landmarks_3d in candidates:
        source_landmarks_2d = {
            name: source_front_points(landmarks_3d[name].reshape(1, 3))[0]
            for name in SUPPORT_NAMES
        }
        src_lm_2d = np.vstack([source_landmarks_2d[name] for name in SUPPORT_NAMES])
        dst_lm_2d = np.vstack([target_landmarks_2d[name] for name in SUPPORT_NAMES])
        if args.alignment_policy == "similarity_support":
            r, scale, t = similarity_2d(src_lm_2d, dst_lm_2d)
        elif args.alignment_policy == "support_similarity_nose_anchor":
            r, scale, _t = similarity_2d(src_lm_2d, dst_lm_2d)
            t = target_landmarks_2d["nose_tip"] - scale * (r @ source_landmarks_2d["nose_tip"])
        elif args.alignment_policy == "nose_cheek_axis":
            r, scale, t = vector_anchor_similarity(
                source_landmarks_2d,
                target_landmarks_2d,
                "left_cheek",
                "right_cheek",
            )
        elif args.alignment_policy == "nose_brow_axis":
            r, scale, t = vector_anchor_similarity(
                source_landmarks_2d,
                target_landmarks_2d,
                "nose_tip",
                "brow_center",
            )
        else:
            raise ValueError(args.alignment_policy)
        vertex_keep = source_region_mask(vertices, landmarks_3d, args.source_fraction_below_nose)

        def evaluate_scale(candidate_scale: float, refined: bool) -> dict:
            if refined:
                candidate_t = (
                    target_landmarks_2d["nose_tip"]
                    - candidate_scale * (r @ source_landmarks_2d["nose_tip"])
                )
            else:
                candidate_t = t
            aligned_landmarks_2d = {
                name: apply_similarity_2d(
                    source_landmarks_2d[name].reshape(1, 2),
                    r,
                    candidate_scale,
                    candidate_t,
                )[0]
                for name in SUPPORT_NAMES
            }
            residuals = np.linalg.norm(
                np.vstack([aligned_landmarks_2d[name] for name in SUPPORT_NAMES]) - dst_lm_2d,
                axis=1,
            )
            aligned_points_2d = apply_similarity_2d(source_points_2d, r, candidate_scale, candidate_t)
            grid = raster_grid(target_points_2d, aligned_points_2d[vertex_keep], args.resolution_mm, args.margin_mm)
            target_mask = points_to_mask(target_points_2d, grid, args.target_dilation_iters)
            source_mask = mesh_to_mask(aligned_points_2d, faces, vertex_keep, grid, args.source_dilation_iters)
            metrics = mask_metrics(source_mask, target_mask, args.resolution_mm)
            metrics.update(
                {
                    "scale_2d": float(candidate_scale),
                    "initial_scale_2d": float(scale),
                    "scale_refinement": args.scale_refinement,
                    "scale_factor": float(candidate_scale / max(scale, 1e-12)),
                    "landmark_rmse_mm": float(np.sqrt(np.mean(residuals**2))),
                    "landmark_median_mm": float(np.median(residuals)),
                    "landmark_max_mm": float(np.max(residuals)),
                    "landmark_residuals_json": json.dumps(
                        {name: float(value) for name, value in zip(SUPPORT_NAMES, residuals)}
                    ),
                    "_source_mask": source_mask,
                    "_target_mask": target_mask,
                    "_grid": grid,
                    "_source_landmarks_2d": aligned_landmarks_2d,
                    "_target_landmarks_2d": target_landmarks_2d,
                }
            )
            metrics["_mask_scale_score"] = (
                50.0 * (1.0 - metrics["dice"]) + 0.2 * metrics["boundary_hd95_mm"]
            )
            return metrics

        if args.scale_refinement == "mask_grid":
            scale_candidates = scale * np.linspace(args.scale_grid_min, args.scale_grid_max, args.scale_grid_steps)
            metrics = min((evaluate_scale(candidate, refined=True) for candidate in scale_candidates), key=lambda row: row["_mask_scale_score"])
        else:
            metrics = evaluate_scale(scale, refined=False)

        metrics.update(
            {
                "method": method,
                "mesh": str(mesh_path),
                "mesh_name": mesh_path.name,
                "orientation": orientation,
                "alignment_policy": args.alignment_policy,
                "support_landmarks": ",".join(SUPPORT_NAMES),
                "source_vertices_in_mask": int(vertex_keep.sum()),
                "source_vertices_total": int(len(vertices)),
                "target_points": int(len(target_points_2d)),
            }
        )
        score = metrics["landmark_rmse_mm"] + 10 * (1 - metrics["dice"]) + 0.1 * metrics["boundary_hd95_mm"]
        if best is None or score < best[0]:
            best = (score, metrics)

    if best is None:
        raise ValueError(f"No landmark-mask overlay candidate for {mesh_path}")
    return best[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mri-patch-mesh", required=True, type=Path)
    parser.add_argument("--mri-patch-metadata", required=True, type=Path)
    parser.add_argument("--photo-mesh-dir", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=["mediapipe", "3ddfa_v2"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*.ply")
    parser.add_argument("--bfm-pkl", type=Path)
    parser.add_argument("--nose-drop-mm", type=float, default=18.0)
    parser.add_argument("--resolution-mm", type=float, default=1.0)
    parser.add_argument("--margin-mm", type=float, default=8.0)
    parser.add_argument("--target-dilation-iters", type=int, default=1)
    parser.add_argument("--source-dilation-iters", type=int, default=1)
    parser.add_argument("--source-fraction-below-nose", type=float, default=0.25)
    parser.add_argument("--allow-lr-swap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scale-refinement", choices=["none", "mask_grid"], default="none")
    parser.add_argument("--scale-grid-min", type=float, default=0.75)
    parser.add_argument("--scale-grid-max", type=float, default=1.6)
    parser.add_argument("--scale-grid-steps", type=int, default=86)
    parser.add_argument(
        "--alignment-policy",
        choices=["similarity_support", "support_similarity_nose_anchor", "nose_cheek_axis", "nose_brow_axis"],
        default="similarity_support",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mri_vertices, _mri_faces = load_geometry(args.mri_patch_mesh)
    metadata = json.loads(args.mri_patch_metadata.read_text(encoding="utf-8"))
    nose_z = float(metadata["landmarks"]["nose_tip"][2])
    target_vertices = mri_vertices[mri_vertices[:, 2] >= nose_z - args.nose_drop_mm]
    target_points_2d = mri_front_points(target_vertices)
    target_landmarks_2d = {
        name: np.asarray(metadata["landmarks"][name], dtype=np.float64)[[0, 2]]
        for name in SUPPORT_NAMES
    }

    tddfa_keypoint_ids = None
    if args.method == "3ddfa_v2":
        if args.bfm_pkl is None:
            raise ValueError("--bfm-pkl is required for 3ddfa_v2")
        tddfa_keypoint_ids = load_3ddfa_keypoint_vertex_ids(args.bfm_pkl)

    rows = []
    for index, mesh_path in enumerate(sorted(args.photo_mesh_dir.glob(args.pattern)), start=1):
        row = evaluate_one(mesh_path, args.method, target_points_2d, target_landmarks_2d, args, tddfa_keypoint_ids)
        overlay_path = args.output_dir / f"{index:02d}_{args.method}_landmark_mask_overlay.png"
        write_overlay(
            row["_source_mask"],
            row["_target_mask"],
            row["_grid"],
            row["_source_landmarks_2d"],
            row["_target_landmarks_2d"],
            overlay_path,
            f"{args.method}: landmark-constrained mask overlay",
            row,
        )
        row["overlay"] = str(overlay_path)
        for key in ["_source_mask", "_target_mask", "_grid", "_source_landmarks_2d", "_target_landmarks_2d"]:
            row.pop(key)
        rows.append(row)

    fieldnames = [
        "method",
        "mesh_name",
        "mesh",
        "orientation",
        "alignment_policy",
        "scale_2d",
        "initial_scale_2d",
        "scale_refinement",
        "scale_factor",
        "support_landmarks",
        "landmark_rmse_mm",
        "landmark_median_mm",
        "landmark_max_mm",
        "dice",
        "iou",
        "source_area_mm2",
        "target_area_mm2",
        "boundary_s2t_median_mm",
        "boundary_s2t_p95_mm",
        "boundary_t2s_median_mm",
        "boundary_t2s_p95_mm",
        "boundary_hd95_mm",
        "boundary_assd_mm",
        "source_vertices_in_mask",
        "source_vertices_total",
        "target_points",
        "landmark_residuals_json",
        "overlay",
    ]
    csv_path = args.output_dir / "landmark_mask_overlay_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    json_path = args.output_dir / "landmark_mask_overlay_metrics.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    run_info = {
        "method": args.method,
        "pattern": args.pattern,
        "mri_patch_mesh": str(args.mri_patch_mesh),
        "mri_patch_metadata": str(args.mri_patch_metadata),
        "support_landmarks": SUPPORT_NAMES,
        "alignment_policy": args.alignment_policy,
        "scale_refinement": args.scale_refinement,
        "resolution_mm": args.resolution_mm,
        "nose_drop_mm": args.nose_drop_mm,
        "source_fraction_below_nose": args.source_fraction_below_nose,
        "csv": str(csv_path),
        "json": str(json_path),
        "n": len(rows),
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
